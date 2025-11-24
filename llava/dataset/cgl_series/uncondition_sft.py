from torch.utils.data import Dataset
import transformers
# from scripts.bos.bos_client import get_url

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
from pathlib import Path
import os

# import 子任务
from .subtask import zb_uncondition_cgl_sft_convert
# from llava.dataset.subtask.zb_subtasks import no_cot_pku_convert
# from llava.dataset/subtask/zb_subtasks/no_cot_pku_convert.py

def select_task(content , cgl_inpainted_imgs_p, cgl_origin_imgs_p):

    return zb_uncondition_cgl_sft_convert.convert(content , cgl_inpainted_imgs_p)

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        # cgl_train_p = '/root/paddlejob/workspace/log/code/others/cgl_dataset/real_train_annotation_xyxy.pkl'
        self.cgl_inpainted_imgs_p = '/root/paddlejob/workspace/log/code/others/cgl_dataset/cgl_inpainting_all'
        self.cgl_origin_imgs_p = '/root/paddlejob/workspace/log/code/others/cgl_pure_imgs'
        try:
            #pku_trainP
            list_data_dict = pkl.load(open(data_path, 'rb'))

        except:
            print("here")

        print(len(list_data_dict))
        
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
                dialog, image = select_task(sources, self.cgl_inpainted_imgs_p, self.cgl_origin_imgs_p)
                # dialog, image = zb_nocot_cgl_convert.convert(sources, self.cgl_inpainted_imgs_p)
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

                data_dict['image'] = image.unsqueeze(0)
                return data_dict
            except Exception as e:
                # raise
                print(e)
                i = np.random.randint(len(self.train_dict))

