from torch.utils.data import Dataset
import transformers
import sys
sys.path.append("/home/share/huadjyin/home/wanghaoran/wanghaoran/project/SEGA/LLaVA_poster_multi_task/LLaVA-main")
# o
# from scripts.bos.bos_client import get_url

import json
import os
import copy
import numpy as np
import requests
from PIL import Image
from llava.dataset.utils.imgproc import  rgba2rgb
import torch
import PIL.Image
import pickle as pkl
import cv2
from llava.dataset.preproc import preprocess_multimodal, preprocess_poster
PIL.Image.MAX_IMAGE_PIXELS = 933120000
local_rank = None

# import 子任务
from llava.dataset.subtask.zb_subtasks import bigposter_align_refine_convert


def select_task(content, score, temp_pred, temp_saliency_map):
    # 这里弄加噪和对齐分布两个convert任务 按比例分配 多训几轮  先都弄成无cot版本 对齐prompt  
    #  noise的无cot 是指不指出问题框  而带上文本  目前看下来 对于对齐分布的  有没有文本意义不大

    # s = pkl.load(open(pk, 'rb'))
    flip = np.random.rand() > 0.5
    flip = False
    select_score = np.random.rand()
    # 目前对齐和随机1:1
    return bigposter_align_refine_convert.convert(content, flip, score, temp_pred)
    
    # if select_score < 0.5:
    #     return zb_refine_11_1_noise_convert.convert_no_score(s, flip, temp_saliency_map)    
    # else:
    #     return zb_refine_11_1_align_convert.convert(s, flip, score, temp_pred)

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
from pathlib import Path
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        '''
        这个是利用两类数据 一个是原始的 当作GT  另一个是现在的 还不是总需要 
        '''
        super(LazySupervisedDataset, self).__init__()
        self.folder_p = data_path
        # list_data_dict = sorted(list(Path(data_path).glob('*.pkl')))
        list_data_dict = pkl.load(open(data_path, 'rb'))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer

        self.data_args = data_args

        scores_p = '/home/share/huadjyin/home/wanghaoran/wanghaoran/project/RALF-main/zb_utils_for_cot_refine/sft_for_datapkl_valid2.pkl'
        self.data_scores = pkl.load(open(scores_p, 'rb'))
        # self.scores_names = self.data_scores.keys()
        # self.train_dict = [ pp for pp in self.train_dict if pp.name in self.scores_names ]
        list_data_dict.sort()
        self.train_dict = list_data_dict
        self.pred_p = '/home/share/huadjyin/home/wanghaoran/wanghaoran/project/SEGA/LLaVA_poster_multi_task/LLaVA-main/aaaa_iccv_re/out/sft_for_datapkl'
        self.old_preds = sorted(list(Path(self.pred_p ).glob('*.pkl')))
        self.saliency_map_folder = '/home/share/huadjyin/home/wanghaoran/wanghaoran/project/RALF-main/bg_sal'

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
                sources = self.train_dict[i] # 这里直接是一个四列表
                
                cur_bg_name = sources[0].split('/')[-1]
                cur_name = f'{i}.pkl'

                scores = self.data_scores[cur_name]

                temp_saliency_map_p = self.saliency_map_folder + '/' + cur_bg_name
                temp_saliency_map = cv2.imread(temp_saliency_map_p,0)

                # 这个因为所有背景都坏上去了 所有我可以开始弄saliency_map

                temp_pred_p = self.pred_p + '/' + cur_name
                temp_pred = pkl.load(open(temp_pred_p, 'rb'))

                dialog, image = select_task(sources, scores, temp_pred, temp_saliency_map)

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
                    clip_image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

                data_dict = preprocess_poster(
                    sources,
                    self.tokenizer,
                    has_image=True)

                if isinstance(i, int):
                    data_dict = dict(input_ids=data_dict["input_ids"][0],
                                    labels=data_dict["labels"][0])

                data_dict['image'] = clip_image.unsqueeze(0)

                return data_dict
            except Exception as e:
                print(e)
                # raise
                i = np.random.randint(len(self.train_dict))

if __name__ == "__main__":
    cur_dataset = LazySupervisedDataset('zb_data/iccv_re_val_train.pkl',None,None)
    for item in cur_dataset:
        print(item)
    pass