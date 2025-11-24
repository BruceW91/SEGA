from torch.utils.data import Dataset
import transformers
from scripts.bos.bos_client import get_url

from torch.utils.data import DataLoader

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

# need_look_list = [
#     12897, 7291, 6062, 12198, 9684, 7171, 12074, 405,
#     4046, 4909, 12653, 2295, 378, 4622, 5543, 3974,
#     9119, 12154, 4534, 10828, 10677
# ]

need_look_list = [
    12897, 7291, 6062, 12198, 9684, 7171, 12074, 405,
    4046, 4909, 12653, 2295, 378, 4622, 5543, 3974,
    9119, 12154, 4534, 10828, 10677, 1945, 13123, 986,
    10376, 13400, 4342, 6430, 1836, 11537, 7775, 2357,
    2483
]


# need_look_list = [378, 1945, 12897, 4046, 4909, 10828, 2483, 12653, 6062, 11537, 986]

# need_look_list = [
#     # 12897, 7291, 6062, 12198, 9684, 7171, 12074, 405,
#     # 4046, 4909, 12653, 2295, 378, 4622, 5543, 3974,
#     # 9119, 12154, 4534, 10828, 10677,
#      1945, 13123, 986,
#     10376, 13400, 4342, 6430, 1836, 11537, 7775, 2357,
#     2483
# ]
# import 子任务
from llava.dataset.subtask import zb_origin_no_pred_continue_cot,zb_no_pred_continue_cot_perturb

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
    weights = [1.0, 0]
    # weights = [0.8, 0.2]
    # 这里面要求我这个 convert 就是专用的dpo convert
    return np.random.choice((zb_origin_no_pred_continue_cot.convert, zb_no_pred_continue_cot_perturb.convert), p=weights )(pk, flip)

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
        print(len(self.list_data_dict),len(list_data_dict))
        rank0_print("Formatting inputs...Skip in lazy mode")
        # self.list_data_dict  = self.list_data_dict [1700:]
        self.tokenizer = tokenizer
        self.data_args = data_args
        rank0_print("Init dataset done")
        self.list_data_dict = [ self.list_data_dict[index] for index in need_look_list]
        print(len(self.list_data_dict))
        print("hh")
        

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
                print(i)
                sources = self.list_data_dict[i]

                dialog, nega_dialog ,image = select_task(sources)   
                # dialog, nega_dialog ,image = debug_task(sources)   
                #这里我直接把正负 dialog 和 image 都给出来  如果负dialoag有就知道类别
                # 这里负累 直接都换成加载pkl的
                
                sources_chosen = preprocess_multimodal(
                    [copy.deepcopy(dialog),],
                    self.data_args)
                data_dict_chosen = preprocess_poster(
                    sources_chosen,
                    self.tokenizer,
                    has_image=True )
                if len(data_dict_chosen['input_ids'][0])>2048:
                    print("give up",i)
                    continue
                
                if nega_dialog is not None:
                    sources_rejected = preprocess_multimodal(
                        [copy.deepcopy(nega_dialog),],
                        self.data_args)

                    data_dict_rejected = preprocess_poster(
                        sources_rejected,
                        self.tokenizer,
                        has_image=True )
                    
                    data_dict = dict(input_ids_chosen=data_dict_chosen["input_ids"][0],
                                labels_chosen=data_dict_chosen["labels"][0],
                                input_ids_rejected=data_dict_chosen["input_ids"][0],    #这里用同一个输入input——ids
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

                data_dict['image'] = image
                # if data_dict["input_ids_chosen"]
                return data_dict
            except Exception as e:
                # raise
                print(e)
                i = np.random.randint(len(self.list_data_dict))

if __name__ == '__main__':

    dset = LazySupervisedDataset('zb_data/infer_total_data.pkl', None, None)
    dataloader = DataLoader(
    dataset=dset,      # 数据集
    batch_size=1,         # 批量大小
    shuffle=False,          # 是否打乱数据
    num_workers=0,         # 用于数据加载的子进程数，0表示在主进程中加载数据）
    )
    len_d = len(dataloader)
    print("hh")
    for batch in dataloader:
        # pass```
        print("jj")
    #在这里进行debug 