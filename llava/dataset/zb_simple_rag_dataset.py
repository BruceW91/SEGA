from torch.utils.data import Dataset
import transformers
# from scripts.bos.bos_client import get_url

import json
import os
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
from .subtask import zb_simple_crello_rag_convert

def select_task(pk,example_pk):

    s = pkl.load(open(pk, 'rb'))
    e_s = pkl.load(open(example_pk, 'rb'))
    flip = np.random.rand() > 0.5
    return zb_simple_crello_rag_convert.convert_and_merge_2_dialog(s, flip, e_s)

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
from pathlib import Path
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        self.folder_p = data_path
        list_data_dict = sorted(list(Path(data_path).glob('*.pkl')))
        rag_path = 'zb_data/train_w_text_query_train_mix_2_train_w_text_query_wo_head_table_between_dataset_indexes_top_k500_norm.pt'
        # rag_path = '/root/paddlejob/workspace/log/code/LLaVA_poster_multi_task/LLaVA-main/crello_train_train_dreamsim_2_crello_train_noaug_60_wo_head_table_between_dataset_indexes_top_k50.pt'
        self.rag_indexes = torch.load(rag_path)
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        valset = list(range(len(list_data_dict)))
        valset = valset[::len(valset)//5][:4]
        self.train_dict = [list_data_dict[x] for x in range(len(list_data_dict)) if x not in valset]
        self.valset = [list_data_dict[x] for x in valset]
        self.data_args = data_args
        rank0_print("Init dataset done")

    def __len__(self):
        return len(self.train_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.train_dict:
            img_tokens = 128#  if 'image' in sample else 0
            length_list.append(5 + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample  in self.train_dict:
            img_tokens = 128#  if 'image' in sample else 0
            length_list.append(5 + img_tokens)
        return length_list

    def __getitem__(self, i):
        while True:
            try:
                sources = self.train_dict[i]
                temp_name = sources.name.split('.')[0]
                temp_rag_index = self.rag_indexes[temp_name][0]
                temp_example_name = temp_rag_index +'.pkl'
                example_pkl_p = os.path.join( self.folder_p, temp_example_name )
                # 这个 select 传进去两个pkl文件的路径就行
                # 后面要改成兼容多个img token 的形式  ntk序列长度感觉需要扩大
                dialog, image, example_img = select_task(sources,example_pkl_p)
                # sources = preprocess_multimodal(
                #     [copy.deepcopy(dialog),],
                #     self.data_args)
                sources = [dialog]
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
                    example_img = expand2square(example_img, tuple(int(x*255) for x in processor.image_mean))
                    example_img = processor.preprocess(example_img, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

                data_dict = preprocess_poster(
                    sources,
                    self.tokenizer,
                    has_image=True)

                if isinstance(i, int):
                    data_dict = dict(input_ids=data_dict["input_ids"][0],
                                    labels=data_dict["labels"][0])

                data_dict['image'] = torch.stack([example_img,image]) 
                # data_dict['image'] = torch.stack([image, example_img]) 
                # data_dict['example_img'] = example_img
                return data_dict
            except Exception as e:
                print(e)
                # raise
                i = np.random.randint(len(self.train_dict))