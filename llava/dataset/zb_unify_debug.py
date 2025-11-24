from torch.utils.data import Dataset
import transformers
from scripts.bos.bos_client import get_url

import json
import copy
import numpy as np
import requests
from PIL import Image
from llava.dataset.utils.imgproc import  rgba2rgb
import torch
import PIL.Image
import pickle as pkl
from llava.dataset.preproc import preprocess_multimodal, preprocess_poster
PIL.Image.MAX_IMAGE_PIXELS = 933120000
local_rank = None

from tqdm import tqdm

from torch.utils.data import DataLoader

from llava.train.unify_dpo_train import DataCollatorForSupervisedDataset, ModelArguments, DataArguments, TrainingArguments

# import 子任务
from llava.model import *
from llava.dataset.subtask import zb_origin_no_pred_continue_cot,zb_no_pred_continue_cot_perturb,zb_unify_cot_perturb,zb_unify_cot_perturb_w_buttons

def merge_parts_of_string(input_str, delimiter,worst_pred):
    # 使用分隔符将字符串分解成列表
    parts = input_str.split(delimiter)
    
    # 确保字符串至少有两个部分
    if len(parts) < 2:
        return "错误：字符串需要至少包含两个部分以进行合并。"
    
    # 合并前两个部分
    first_two_combined = delimiter.join(parts[:2])
    
    # 合并剩余的所有部分
    # remaining_parts_combined = delimiter.join(parts[2:])
    
    # 将前两个部分与剩余部分合并
    result = first_two_combined + delimiter + worst_pred
    
    return result


def select_task(pk, inlayer = -1):
    flip = np.random.rand() > 0.5
    # 这里面要求我这个 convert 就是专用的dpo convert
    return zb_unify_cot_perturb_w_buttons.convert(pk, flip, inlayer)

def debug_task(pk):
    flip = np.random.rand() > 0.5
    return zb_no_pred_continue_cot_perturb.convert(pk, flip)

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def check_data(list_data_dict):
    new_list_data_dict = []
    list_len = len(list_data_dict)
    for item in list_data_dict:
        if len(item[1]) == 0:
            continue
        else:
            new_list_data_dict.append(item)
    return new_list_data_dict


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()

        try:
            with open(data_path, 'rb') as file:
                list_data_dict = pkl.load(file)  

        except:
            url = get_url(data_path)
            response = requests.get(url)
            list_data_dict = pkl.loads(response.content)
            list_data_dict = [(a,b,c,d) for a,b,c,d in list_data_dict if len(b) > 0]

        self.list_data_dict  = check_data(list_data_dict)
        # self.problem_list = [2942, 3458, 11652, 4295, 5873, 4047, 1459, 696, 4271, 11636, 1705, 9686]
        # self.cor_layers = [6, 8, 5, 0, 3, 3, 0, 3, 1, 1, 2, 0]
        # self.list_data_dict = [ temp_l[index] for index in problem_list ]
        print(len(self.list_data_dict),len(list_data_dict))
        rank0_print("Formatting inputs...Skip in lazy mode")
        # self.list_data_dict  = self.list_data_dict [1700:]
        self.tokenizer = tokenizer
        self.data_args = data_args
        rank0_print("Init dataset done")
        # self.list_data_dict = self.list_data_dict[:280]
        
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for _, sample, _, b_ in self.list_data_dict:
            img_tokens = 128#  if 'image' in sample else 0
            length_list.append(len(sample) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for _, sample, _ , b_  in self.list_data_dict:
            img_tokens = 128#  if 'image' in sample else 0
            length_list.append(len(sample) + img_tokens)
        return length_list

    def __getitem__(self, i):
        while True:
            try:
                # print(i)
                i = 33
                # index = self.problem_list[i]
                sources = self.list_data_dict[index]
                # in_layer = self.cor_layers[i]

                dialog, nega_dialog ,image,done = select_task(sources)   
                
                sources_chosen = preprocess_multimodal(
                    [copy.deepcopy(dialog),],
                    self.data_args)
                data_dict_chosen = preprocess_poster(
                    sources_chosen,
                    self.tokenizer,
                    has_image=True )
                
                sources_rejected = preprocess_multimodal(
                    [copy.deepcopy(nega_dialog),],
                    self.data_args)

                data_dict_rejected = preprocess_poster(
                    sources_rejected,
                    self.tokenizer,
                    has_image=True )
                
                data_dict = dict(input_ids_chosen=data_dict_chosen["input_ids"][0],
                            labels_chosen=data_dict_chosen["labels"][0],
                            input_ids_rejected=data_dict_rejected["input_ids"][0],    #这里用同一个输入input——ids   现在改了
                            labels_rejected=data_dict_rejected["labels"][0],dpo_task = True, data_index = index, done_layer = done)
                if data_dict_chosen["input_ids"][0].shape < data_dict_rejected["input_ids"][0].shape:
                    print("look")         
                if data_dict_chosen["input_ids"][0].shape != data_dict_chosen["labels"][0].shape:
                    raise Exception("input_ids_chosen.shape != labels_chosen.shape",i)
                if data_dict_rejected["input_ids"][0].shape != data_dict_rejected["labels"][0].shape:
                    raise Exception("input_ids_chosen.shape != labels_chosen.shape",i)

                #下面上图像处理逻辑  
                processor = self.data_args.image_processor
                image = rgba2rgb(image)
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

                data_dict['image'] = image
                return data_dict
            except Exception as e:
                # raise
                print(e)
                i = np.random.randint(len(self.list_data_dict))

if __name__ == '__main__':

    attn_implementation = 'flash_attention_2'
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    bnb_model_from_pretrained_args = {}
    model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else torch.float16),
                **bnb_model_from_pretrained_args
            )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    tokenizer.pad_token = tokenizer.unk_token

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    dset = LazySupervisedDataset('zb_data/dpo_1w_plus.pkl', tokenizer, data_args)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    dataloader = DataLoader(
    dataset=dset,      # 数据集
    batch_size=1,         # 批量大小
    shuffle=False,          # 是否打乱数据
    num_workers=1,  
    collate_fn=data_collator       # 用于数据加载的子进程数，0表示在主进程中加载数据）
    )
    epoch = 1
    for idx,data in enumerate(tqdm(dataloader)):
        print(idx,'\n')
