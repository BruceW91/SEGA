import argparse
import os
import pdb
from copy import deepcopy
from typing import Any

from llava.mm_utils import get_model_name_from_path
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llava.model.builder import load_pretrained_model,load_pretrained_model_no_merge
from llava.train.train import make_supervised_data_module

from torch.utils.data import DataLoader

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers

from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType

def _project(current_full_grads, projected_grads, projectors, model_id, proj_dim):
    device = torch.device('cuda:0')
    current_full_grads = torch.stack(current_full_grads).to(torch.float16).to(device)
    for i, projector in enumerate(projectors):
        current_projected_grads = projector.project(
            current_full_grads, model_id=model_id)
        projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

def _save(projected_grads, output_dirs, proj_dim, count):
    for dim in proj_dim:
        if len(projected_grads[dim]) == 0:
            continue
        projected_grads[dim] = torch.cat(projected_grads[dim])

        output_dir = output_dirs[dim]
        outfile = os.path.join(output_dir, f"grads-{count}.pt")
        torch.save(projected_grads[dim], outfile)
        print(
            f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
        projected_grads[dim] = []

def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params

def obtain_gradients(model, batch):
    """ obtain gradients. """
    loss = model(**batch).loss
    loss.backward()
    # print("hh")
    vectorized_grads = torch.cat(
        [p.grad.view(-1).cpu() for p in model.parameters() if p.grad is not None])
    return vectorized_grads