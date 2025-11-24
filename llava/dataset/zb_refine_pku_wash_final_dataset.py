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
import cv2
from .preproc import preprocess_multimodal, preprocess_poster
PIL.Image.MAX_IMAGE_PIXELS = 933120000
local_rank = None

# import 子任务
from .subtask import zb_refine_final_noise_pku_convert, zb_refine_final_align_pku_convert

def select_task(content, score, temp_pred, temp_saliency_map, img_folder_p):
    # 这里弄加噪和对齐分布两个convert任务 按比例分配 多训几轮  先都弄成无cot版本 对齐prompt  
    #  noise的无cot 是指不指出问题框  而带上文本  目前看下来 对于对齐分布的  有没有文本意义不大

    # s = pkl.load(open(pk, 'rb'))
    # flip = np.random.rand() > 0.5
    select_score = np.random.rand()
    # 目前对齐和随机1:1
    if select_score < 0.5:
        return zb_refine_final_noise_pku_convert.convert(content, img_folder_p, temp_saliency_map)
    else:
        return zb_refine_final_align_pku_convert.convert(content, score, temp_pred, img_folder_p)

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
        pku_train_p = '/root/paddlejob/workspace/log/code/pku_all_need/pku_dataset/pku_train_dict.pkl'
        self.pku_inpainted_imgs_p = '/root/paddlejob/workspace/log/code/pku_all_need/pku_dataset/input'
        pku_data = pkl.load(open(pku_train_p, 'rb'))
        imgs_names = os.listdir(self.pku_inpainted_imgs_p) 

        scores_p = '/root/paddlejob/workspace/log/code/RALF-main/zb_utils_for_cot_refine/pku_sft_800_in_trainpkl.pkl'
        # scores_p = '/root/paddlejob/workspace/log/code/RALF-main/zb_utils_for_cot_refine/full_simplesft_17e_test_in_alltrain_rpkl.pkl'
        self.data_scores = pkl.load(open(scores_p, 'rb'))
        self.scores_names = self.data_scores.keys()
        self.scores_names = [ name.split(".")[0] +'.png' for name in self.scores_names ]

        list_data_dict = [ (k,v) for k, v in pku_data.items() if k in imgs_names and k in self.scores_names]

        self.tokenizer = tokenizer
        valset = list(range(len(list_data_dict)))
        valset = valset[::len(valset)//5][:4]
        self.train_dict = [list_data_dict[x] for x in range(len(list_data_dict)) if x not in valset]
        self.valset = [list_data_dict[x] for x in valset]
        self.data_args = data_args

        self.pred_p = '/root/paddlejob/workspace/log/code/LLaVA_poster_multi_task/LLaVA-main/infer_out/pku_sft_800_in_trainpkl'
        self.saliency_map_folder = 'llava/dataset/zb_refine_noise_utils/pku_union_sal'

        wash_768_path = '/root/paddlejob/workspace/log/code/RALF-main/zb_utils_data/10_26_pku_wash1411_wread_bad_names_final.pkl'
        # wash_768_path = '/root/paddlejob/workspace/log/code/RALF-main/zb_utils_data/10_23_pku_wash_bad_names_final.pkl'
        wash_768_names = pkl.load(open(wash_768_path, 'rb'))
        self.train_dict = [ pp for pp in self.train_dict if pp[0].replace(".png", ".pkl") not in wash_768_names ]    
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
                temp_name = sources[0].split('.')[0]
                # temp_name
                scores = self.data_scores[temp_name+'.pkl']

                temp_saliency_map_p = self.saliency_map_folder + '/' + temp_name + '.png'
                temp_saliency_map = PIL.Image.open(temp_saliency_map_p)

                temp_pred_p = self.pred_p + '/' + temp_name + '.pkl'
                temp_pred = pkl.load(open(temp_pred_p, 'rb'))

                dialog, image = select_task(sources, scores, temp_pred, temp_saliency_map, self.pku_inpainted_imgs_p)

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
                # data_dict['image'] = torch.stack([example_img,image]) 

                return data_dict
            except Exception as e:
                print(e)
                # raise
                i = np.random.randint(len(self.train_dict))