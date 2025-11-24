"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import json
import pickle


# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.utils.quantization_config import QuantizationMethod





import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
# from ..eval.run_llava import *

import numpy as np
import cv2
from PIL import Image
import copy
import json
from ..utils import get_url, load_remote_json

import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

from accelerate import Accelerator

def rgba2rgb(png):
    png = png.convert('RGBA')
    background = Image.new('RGBA', png.size, (255, 255, 255))

    alpha_composite = Image.alpha_composite(background, png).convert("RGB")
    return alpha_composite
def process_image(img, out_size=1080):
    img = img#np.array(rgba2rgb(img))
    size = img.shape[:2]
    image = np.zeros([max(size),max(size),3])
    if size[0] > size[1]:
        diff = (size[0]-size[1]) // 2
        image[:,diff:diff+size[1],:] = img
    else:
        diff = (size[1]-size[0]) // 2
        image[diff:diff+size[0],:,:] = img
    return np.array(cv2.resize(image, (out_size,out_size), interpolation=cv2.INTER_NEAREST),np.int32)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True): # 真随机
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality
        self.iters = 0

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality: # Enter
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)
    

import importlib.util

def dynamic_import_function(module_path, function_name):
    # 加载模块
    name = module_path.split('/')[-1].split('.py')[0]
    spec = importlib.util.spec_from_file_location(f"llava.train.val.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取函数对象
    function = getattr(module, function_name)
    return function
def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


class LLaVATrainer_DPO(Trainer):
    def __init__(self, model, tokenizer, args, local_rank, data_module, reference_model, loss_type):
        super().__init__(model=model, tokenizer=tokenizer, args=args, **data_module)
        self.local_rank = local_rank
        self.reference_model = reference_model.to(args.device)
        self.loss_type = loss_type
        # self.reference_model = reference_model.load_state_dict(model.state_dict())
        # self.loss_type = loss_type
        # self.reference_model = None
        data_path = args.validation_data
        assert data_path is not None, "未指定validation dataset!"
        assert args.validation_func is not None, "未指定validation 函数!"
        self.validation_func = dynamic_import_function(args.validation_func, 'evaluate')
        try:
            self.jsons_dicts = json.load(open(data_path, "r"))
        except:
            url = get_url(data_path)
            self.jsons_dicts = load_remote_json(url)
        if self.local_rank == 0:
            print(self.args.do_eval)
        for _ in self.jsons_dicts[:20]:
            pass
        tb_writer = None
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writer = tb_writer


    
    def preference_loss(self, policy_chosen_logps: torch.FloatTensor,
                        policy_rejected_logps: torch.FloatTensor,
                        reference_chosen_logps: torch.FloatTensor,
                        reference_rejected_logps: torch.FloatTensor,
                        beta: float,
                        label_smoothing: float = 0.0,
                        ipo: bool = False,
                        reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
            ipo: If True, use the IPO loss instead of the DPO loss.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        # pi_logratios = pi_logratios.to(self.accelerator.device)
        # ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing)
                - F.logsigmoid(-beta * logits) * label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.loss_type == "dpop":
            lambda_coef = 50.0
            logits_penalty = reference_chosen_logps - policy_chosen_logps
            # losses = (
            #     -(F.logsigmoid(beta * logits) - lambda_coef * max(0, logits_penalty)) * (1 - label_smoothing)
            #     -(F.logsigmoid(-beta * logits) - lambda_coef * max(0, logits_penalty)) * label_smoothing
            # )
            losses = (
                -(F.logsigmoid(beta * logits) - lambda_coef *  logits_penalty.clamp(min=0)) * (1 - label_smoothing)
                -(F.logsigmoid(-beta * logits) - lambda_coef *  logits_penalty.clamp(min=0)) * label_smoothing
            )
        
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards


        # pi_logratios = policy_chosen_logps - policy_rejected_logps
        # ref_logratios = reference_chosen_logps - reference_rejected_logps


    def _get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)    


    # def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], flag=0) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    #     """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
    #        We do this to avoid doing two forward passes, because it's faster for FSDP.
    #     """
    #     # concatenated_batch = concatenated_inputs(batch)

    #     inputs_cp = copy.deepcopy(batch)
    #     if flag == 0:
    #         labels = inputs_cp.pop("labels")
    #     else:
    #         labels = inputs_cp["labels"]
    #     outputs = model(**inputs_cp)
    #     logits = outputs.logits.to(torch.float32)

    #     logps = self._get_batch_logps(logits = logits, labels = labels, average_log_prob = False)
    #     # print('logps', type(logps))   

    #     return logps, outputs
    

    def zb_concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], c_labels, r_labels) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # concatenated_batch = concatenated_inputs(batch)

        inputs_cp = copy.deepcopy(batch)
        # if inputs_cp['input_ids'].shape[1] == 
        outputs = model(**inputs_cp)
        logits = outputs.logits.to(torch.float32)

        if  logits.shape[:-1] != c_labels.shape:
            print("hh")

        # outputs = model(**inputs_cp)

        c_logps = self._get_batch_logps(logits = logits, labels = c_labels, average_log_prob = False)
        r_logps = self._get_batch_logps(logits = logits, labels = r_labels, average_log_prob = False)
        # print('logps', type(logps))   

        return c_logps, r_logps, outputs


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        现在已经做好了格式的统一

        Subclass and override for custom behavior.
        """
        #   input_ids
        #   labels
        #   attention_mask
        # self.iters += 1

        batch_flags = inputs['dpo_flags']
        has_true = any(batch_flags)

        chosen_inputs = {}
        chosen_inputs['input_ids'] = inputs['input_ids_chosen']
        chosen_inputs['labels'] = inputs['labels_chosen']
        chosen_inputs['attention_mask'] = inputs['attention_mask_chosen']
        chosen_inputs['images'] = inputs['images']
        
        rejected_inputs = {}
        rejected_inputs['input_ids'] = inputs['input_ids_rejected'][batch_flags]
        rejected_inputs['labels'] = inputs['labels_rejected'][batch_flags]
        rejected_inputs['attention_mask'] = inputs['attention_mask_rejected'][batch_flags]
        rejected_inputs['images'] = inputs['images'][batch_flags]
        
        # print(type(chosen_inputs['input_ids']), type(chosen_inputs['labels']))
        if inputs['input_ids_chosen'].dtype != torch.int64 or inputs['labels_chosen'].dtype != torch.int64:
            chosen_inputs['input_ids'] = chosen_inputs['input_ids'].to(torch.int64)
            chosen_inputs['labels'] = chosen_inputs['labels'].to(torch.int64)
            chosen_inputs['attention_mask'] = chosen_inputs['attention_mask'].to(torch.int64)
            # rejected_input

        #下面可以根据batch_flags 这个mask 简化计算  查看一下true false  我这里想要 dpo的是true 这样一下子取出来  现在就没问题

        policy_chosen_logps, policy_rejected_logps ,outputs = self.zb_concatenated_forward(model, chosen_inputs, inputs['new_labels_chosen'], inputs['new_labels_rejected'])

        if has_true:

            new_chosen_inputs = {}
            new_chosen_inputs['input_ids'] = inputs['input_ids_chosen'][batch_flags]
            new_chosen_inputs['labels'] = inputs['labels_chosen'][batch_flags]
            new_chosen_inputs['attention_mask'] = inputs['attention_mask_chosen'][batch_flags]
            new_chosen_inputs['images'] = inputs['images'][batch_flags]

            with torch.no_grad():
                # model.load_state_dict(self.reference_dict, strict = False)
                reference_chosen_logps, reference_rejected_logps, _ = self.zb_concatenated_forward(self.reference_model, copy.deepcopy(chosen_inputs), inputs['new_labels_chosen'], inputs['new_labels_rejected'])  # 这里为了兼容 multitask 跑全batch的推理了
            
            loss_kwargs = {'beta': 0.2, 'reference_free': False, 'label_smoothing': 0, 'ipo': False}

            
            losses, chosen_rewards, rejected_rewards = self.preference_loss(
                    policy_chosen_logps[batch_flags], policy_rejected_logps[batch_flags], reference_chosen_logps[batch_flags], reference_rejected_logps[batch_flags], **loss_kwargs)
            losses = losses.mean()

            self.log({"chosen_rewards": chosen_rewards.mean().item(), "rejected_rewards": rejected_rewards.mean().item()})
        else:
            losses = torch.tensor(0, dtype=torch.float32)   
        # # Save past state if it exists
        # # TODO: this needs to be fixed and made cleaner later.  
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        auto_reg = outputs["loss"]
        self.log({"autoreg": auto_reg.item(), "dpoloss": losses.item()})
        # self.log({"chosen_rewards": chosen_rewards.mean().item(), "rejected_rewards": rejected_rewards.mean().item()})

        # print('autoreg:', auto_reg.item(),'dpoloss:',losses.item())
        # effective_weight = 1
        # effective_weight = 0
        # if losses.item() < 1 :
        #     effective_weight = 0.0 
        total = auto_reg + losses * 0.01

        # total = losses3        # total = auto_reg + losses * 0.01
        # total = losses 
        # total = auto_reg + losses 
        # total = auto_reg + losses * effective_weight
            
        # return (losses, outputs) if return_outputs else losses
        return (total, outputs) if return_outputs else total
        
    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths # 每一条语料的单词数量
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            # return SequentialSampler(self.train_dataset)
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.
        
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
            super(LLaVATrainer_DPO, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer_DPO, self)._save(output_dir, state_dict)

    def evaluate(
        self,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        self.model.eval()
        if self.tb_writer is None:
            self._init_summary_writer(self.args)
        if self.local_rank == 0:
            self.validation_func(self, ignore_keys, metric_key_prefix)
        self.model.train()
