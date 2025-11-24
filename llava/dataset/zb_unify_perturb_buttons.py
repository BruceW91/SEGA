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

from torch.utils.data import DataLoader

# import 子任务   vllava/dataset/subtask/zb_unify_cot_perturb_buttons.py
from .subtask import zb_origin_no_pred_continue_cot,zb_no_pred_continue_cot_perturb,zb_unify_cot_perturb,zb_unify_cot_perturb_w_buttons,zb_unify_cot_perturb_buttons
# llava/dataset/zb_unify_perturb_buttons.py
import random

def zb_perturb_bbox(bbox, translate_ratio=0.1, scale_ratio=0.1):
    """
    对0-1归一化的bbox进行平移和缩放的扰动。
    
    参数:
    bbox: list or tuple, 形如[x_min, y_min, x_max, y_max]的归一化bbox，其中每个值都在[0, 1]之间。
    translate_ratio: float, 平移比例，取值范围建议在0到1之间，表示bbox可以在其原始位置基础上最大平移自身宽度或高度的translate_ratio倍。
    scale_ratio: float, 缩放比例，可以是正数也可以是负数，表示bbox可以在其原始尺寸基础上最大放大或缩小scale_ratio倍。
    
    返回:
    list, 扰动后的bbox。
    """
    # 解构bbox
    x_min, y_min, x_max, y_max = bbox
    
    # 计算bbox的宽度和高度
    width = x_max - x_min
    height = y_max - y_min
    
    # 计算平移量
    translate_x = (x_max - x_min) * translate_ratio
    translate_y = (y_max - y_min) * translate_ratio
    
    # 随机决定平移的方向（正向或负向）
    random_sign_x = -1 if random.random() < 0.5 else 1
    random_sign_y = -1 if random.random() < 0.5 else 1
    dx = translate_x * random_sign_x
    dy = translate_y * random_sign_y
    
    # 应用平移
    new_x_min = max(0, x_min + dx)
    new_y_min = max(0, y_min + dy)
    new_x_max = min(1, x_max + dx)
    new_y_max = min(1, y_max + dy)
    
    # 计算缩放因子
    scale_factor = 1 + scale_ratio * (2 * random.random() - 1)  # 保证了scale_factor在[1-scale_ratio, 1+scale_ratio]之间
    
    # 应用缩放
    new_width = width * scale_factor
    new_height = height * scale_factor
    
    # 确保缩放后bbox仍在[0, 1]范围内
    new_x_max = min(1, new_x_min + new_width)
    new_y_max = min(1, new_y_min + new_height)
    
    # return [new_x_min, new_y_min, new_x_max, new_y_max]
    return [round(val, 3) for val in [new_x_min, new_y_min, new_x_max, new_y_max]], scale_ratio

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
    # 这里面要求我这个 convert 就是专用的dpo convert
    return zb_unify_cot_perturb_buttons.convert(pk, flip)

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
        print(len(self.list_data_dict),len(list_data_dict))
        rank0_print("Formatting inputs...Skip in lazy mode")
        # self.list_data_dict  = self.list_data_dict [1700:]
        self.tokenizer = tokenizer
        self.data_args = data_args
        rank0_print("Init dataset done")
        
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
        for _, sample, _, b_  in self.list_data_dict:
            img_tokens = 128#  if 'image' in sample else 0
            length_list.append(len(sample) + img_tokens)
        return length_list

    def __getitem__(self, i):
        while True:
            try:
                sources = self.list_data_dict[i]

                dialog, nega_dialog ,image, done = select_task(sources)   
                
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
                            input_ids_rejected=data_dict_rejected["input_ids"][0],    #这里用同一个输入input——ids
                            labels_rejected=data_dict_rejected["labels"][0],data_index = i, done_layer = done)
                
                if data_dict_chosen["input_ids"][0].shape < data_dict_rejected["input_ids"][0].shape:
                    print("look")                    

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

    dset = LazySupervisedDataset('zb_data/infer_total_data.pkl', None, None)
    dataloader = DataLoader(
    dataset=dset,      # 数据集
    batch_size=10,         # 批量大小
    shuffle=True,          # 是否打乱数据
    num_workers=0,         # 用于数据加载的子进程数，0表示在主进程中加载数据）
    )