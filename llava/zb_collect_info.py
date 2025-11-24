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

from llava.zb_utils_for_collect_grad_info import _project, _save, get_number_of_params, obtain_gradients

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

import pickle as pkl
'''
要改dataset 输出文件名字 只有batchsize 为1 才能每个poster 一个梯度   都存下来，最后在做平均
'''

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        names = [instance['name'] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        batch['name'] = names

        return batch
    
def prepare_batch(batch, dtype, device=torch.device("cuda:0"),):
    """ Move the batch to the device. """
    for key in batch:
        if 'image' in key:
            batch[key] = batch[key].to(device).to(dtype)
        elif 'name' in key:
            pass
        else:
            batch[key] = batch[key].to(device)

def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector

@dataclass
class DataArguments:
    dataset: str = 'llava/dataset/zb_simple_crello_cot_for_grad_info.py'
    data_path: str = field(default='/root/paddlejob/workspace/log/code/zcrellow_train_underlay',
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str]  = field(default=None)
    image_aspect_ratio: str = 'pad'

def load_model(args):
    # disable_torch_init()
    
    tokenizer, model, image_processor, context_len = load_pretrained_model_no_merge(
        args.model_path, args.model_base, args.model_name
    )
    
    args.tokenizer = tokenizer
    args.model = model
    args.image_processor = image_processor

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True

    return model,image_processor,tokenizer


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, default='llava-15')
parser.add_argument("--lora", type=str, default='checkpoints/full_train_2048_wontk_17e_simple/checkpoint-2000')
args = parser.parse_args()

model_path = args.lora # wjh
model_base = args.base # wjh
conv_mode = 'poster_llava_general_wjh_0407'

args = type('Args', (), {
    "model_path": model_path,
    "model_base": model_base,
    "model_name": get_model_name_from_path(model_path) +'llava_lora',
    "conv_mode": conv_mode,
    "sep": ",",
    "temperature": 0.2,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 4096,
    'json_dict':''
})()

model,image_processor,tokenizer = load_model(args)

parser = transformers.HfArgumentParser(
    DataArguments)
data_args = parser.parse_args_into_dataclasses()[0]
data_args.image_processor = image_processor
data_args.mm_use_im_start_end = False

data_module = make_supervised_data_module(tokenizer=tokenizer,
                                            data_args=data_args)
cur_data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

dataloader = DataLoader(data_module['train_dataset'],batch_size=1,collate_fn=cur_data_collator)  # When getting gradients, we only do this single batch process)
# dataloader = DataLoader(data_module['train_dataset'],batch_size=4,  # When getting gradients, we only do this single batch process
#                             collate_fn=data_module['data_collator'])

version_name = "full_sft_crello_17e_from_llava_2000_simple_no_cot"
# version_name = "full_sft_crello_17e_from_llava_500"
# version_name = "full_sft_crello_40e_from_llava_5400"
output_dir_f = 'zb_data/grad_info/'
output_dir = os.path.join(output_dir_f, version_name)

device = next(model.parameters()).device
dtype = next(model.parameters()).dtype

projector = get_trak_projector(device)
number_of_params = get_number_of_params(model)
count = 0

project_interval = 64
save_interval = 640
model_id = 0  # model_id is used to draft the random seed for the projectors
block_size = 128  # fixed block size for the projectors
projector_batch_size = 64  # batch size for the projectors
torch.random.manual_seed(0)  # set the random seed for torch
proj_dim = [8192*2]

# initialize a project for each target projector dimension
projectors = []
for dim in proj_dim:
    proj = projector(grad_dim=number_of_params,
                        proj_dim=dim,
                        seed=0,
                        proj_type=ProjectionType.rademacher,
                        device=device,
                        dtype=dtype,
                        block_size=block_size,
                        max_batch_size=projector_batch_size)
    projectors.append(proj)

full_grads = []  # full gradients
projected_grads = {dim: [] for dim in proj_dim}  # projected gradients

# set up a output directory for each dimension
#不理解
output_dirs = {}
for dim in proj_dim:
    output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
    output_dirs[dim] = output_dir_per_dim
    os.makedirs(output_dir_per_dim, exist_ok=True)
name_list = []
for batch in tqdm(dataloader, total=len(dataloader)):
    '''
    现在可以设计成 1000个存一下 一会再写一个match 函数就可以了
    '''
    count += 1
    prepare_batch(batch,dtype)

    name = batch.pop('name')
    name_list.append(name)

    vectorized_grads = obtain_gradients(model, batch)
    # add the gradients to the full_grads
    full_grads.append(vectorized_grads)
    model.zero_grad()

    if count % project_interval == 0:
        _project(full_grads, projected_grads, projectors, model_id, proj_dim)
        full_grads = []
    if count % save_interval == 0:
        _save(projected_grads, output_dirs, proj_dim, count)

    print("count:", count)

if len(full_grads) > 0:
    _project(full_grads, projected_grads, projectors, model_id, proj_dim)
    full_grads = []

for dim in proj_dim:
    _save(projected_grads, output_dirs, proj_dim, count)

dst_name_p =  os.path.join(output_dir,"name_list.pkl")   # output_dir
# 最后存一下namelist 的pkl
pkl.dump(name_list, open(dst_name_p,'wb'))
print("hh")