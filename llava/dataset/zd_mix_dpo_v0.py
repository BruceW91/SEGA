from torch.utils.data import Dataset
import transformers
from scripts.bos.bos_client import get_url

import json
import copy
import numpy as np
import requests
from PIL import Image
from .utils.imgproc import  rgba2rgb
import torch
import PIL.Image
import pickle as pkl
from .preproc import preprocess_multimodal, preprocess_poster
PIL.Image.MAX_IMAGE_PIXELS = 933120000
local_rank = None

# import 子任务
from .subtask import zb_origin_no_pred_continue_cot,zb_no_pred_continue_cot_perturb


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


def select_task(pk):
    flip = np.random.rand() > 0.5
    weights = [0, 1.0]
    # 这里面要求我这个 convert 就是专用的dpo convert
    return np.random.choice((zb_origin_no_pred_continue_cot.convert, zb_no_pred_continue_cot_perturb.convert), p=weights )(pk, flip)

def debug_task(pk):
    flip = np.random.rand() > 0.5
    return zb_no_pred_continue_cot_perturb.convert(pk, flip)

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
            with open(data_path, 'rb') as file:
                self.list_data_dict = pkl.load(file)  
        except:
            url = get_url(data_path)
            response = requests.get(url)
            list_data_dict = pkl.loads(response.content)
            list_data_dict = [(a,b,c,d) for a,b,c,d in list_data_dict if len(b) > 0]

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        rank0_print("Init dataset done")
        
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for _, sample, _,in self.list_data_dict:
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
                sources = self.list_data_dict[i]

                dialog, nega_dialog ,image = select_task(sources)   
                # dialog, nega_dialog ,image = debug_task(sources)   
                #这里我直接把正负 dialog 和 image 都给出来  如果负dialoag有就知道类别
                
                sources_chosen = preprocess_multimodal(
                    [copy.deepcopy(dialog),],
                    self.data_args)
                data_dict_chosen = preprocess_poster(
                    sources_chosen,
                    self.tokenizer,
                    has_image=True)
                
                if nega_dialog is not None:
                    sources_rejected = preprocess_multimodal(
                        [copy.deepcopy(nega_dialog),],
                        self.data_args)

                    data_dict_rejected = preprocess_poster(
                        sources_rejected,
                        self.tokenizer,
                        has_image=True)
                    
                    data_dict = dict(input_ids_chosen=data_dict_chosen["input_ids"][0],
                                labels_chosen=data_dict_chosen["labels"][0],
                                input_ids_rejected=data_dict_rejected["input_ids"][0],   #改了    #扰动导致编码长度不一样了  ！！！
                                # input_ids_rejected=data_dict_chosen["input_ids"][0],   #改了    #扰动导致编码长度不一样了  ！！！
                                labels_rejected=data_dict_rejected["labels"][0],dpo_task = True)
                    if data_dict_chosen["input_ids"][0].shape < data_dict_rejected["input_ids"][0].shape:
                        print("look")
                else:
                    # data_dict_rejected = {'input_ids': [[None]], 'labels': [[None]]}
                    data_dict = dict(input_ids=data_dict_chosen["input_ids"][0],       
                                labels=data_dict_chosen["labels"][0],dpo_task =False)   


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

                # if isinstance(i, int):
                #     data_dict = dict(input_ids=data_dict["input_ids"][0],
                #                     labels=data_dict["labels"][0])

                data_dict['image'] = image
                return data_dict
            except Exception as e:
                # raise
                print(e)
                i = np.random.randint(len(self.list_data_dict))

