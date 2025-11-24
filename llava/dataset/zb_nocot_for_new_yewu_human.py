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
from llava.dataset.preproc  import preprocess_multimodal, preprocess_poster
PIL.Image.MAX_IMAGE_PIXELS = 933120000
local_rank = None

from torch.utils.data import DataLoader
from tqdm import tqdm
from llava.model import *

from llava.train.train import DataCollatorForSupervisedDataset, ModelArguments, DataArguments, TrainingArguments


# import 子任务
# from .subtask import no_pred_continue_cot
from llava.dataset.subtask  import no_pred_continue_cot_for_0530,zb_continue_nocot_for_yewu,zb_continue_nocot_for_new_yewu_human

def select_task(pk):
    flip = np.random.rand() > 0.5
    return np.random.choice((zb_continue_nocot_for_new_yewu_human.convert,))(pk, flip)

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        try:
            # list_data_dict = json.load(open(data_path, "r"))
            with open(data_path, 'rb') as file:
                self.list_data_dict = pkl.load(file)  
            # list_data_dict = json.load(open(data_path, "r"))
        except:
            url = get_url(data_path)
            response = requests.get(url)
            list_data_dict = pkl.loads(response.content)
            list_data_dict = [(a,b,c,d) for a,b,c,d in list_data_dict if len(b) > 0]
        self.list_data_dict = [a for a in self.list_data_dict if a != None]
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer

        self.data_args = data_args
        rank0_print("Init dataset done")
        
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for _, sample, _ in self.list_data_dict:
            img_tokens = 128#  if 'image' in sample else 0
            length_list.append(len(sample) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for _, sample, _  in self.list_data_dict:
            img_tokens = 128#  if 'image' in sample else 0
            length_list.append(len(sample) + img_tokens)
        return length_list

    def __getitem__(self, i):
        while True:
            try:
                # i = 127
                sources = self.list_data_dict[i]
                dialog, image = select_task(sources)
                sources = preprocess_multimodal(
                    [copy.deepcopy(dialog),],
                    self.data_args)
                
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


                data_dict = preprocess_poster(
                    sources,
                    self.tokenizer,
                    has_image=True)
                
                if isinstance(i, int):
                    data_dict = dict(input_ids=data_dict["input_ids"][0],
                                    labels=data_dict["labels"][0])

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

    dset = LazySupervisedDataset('zb_data/for_yewu/inpainting_and_wash_series/merge_no_underlay_7_18_2_human.pkl', tokenizer, data_args)


    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    dataloader = DataLoader(
    dataset=dset,      # 数据集
    batch_size=3,         # 批量大小
    shuffle=False,          # 是否打乱数据
    num_workers=1,  
    collate_fn=data_collator       # 用于数据加载的子进程数，0表示在主进程中加载数据）
    )
    epoch = 1
    for idx,data in enumerate(tqdm(dataloader)):
        print(idx,'\n')