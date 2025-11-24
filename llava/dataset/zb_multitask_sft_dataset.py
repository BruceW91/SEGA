import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
# from pycocotools import mask
from transformers import CLIPImageProcessor
import transformers


# from llava.dataset.zb_crello_simple_cot_for_rag import LazySupervisedDataset/
from llava.dataset.zb_multitask_sft_utils.simple_crello_sft_dataset import LazySupervisedDataset
from llava.dataset.zb_multitask_sft_utils.vqa_dataset import VQADataset


# 寻找特定值之后的第一个数值的位置

class HybridDataset(torch.utils.data.Dataset):
    '''
    这里我目前组织成 海报sft 和  VQA 两个任务吧先
    '''

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        samples_per_epoch=500 * 8 * 5,
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="vqa||poster",
        sample_rate=[1, 19],
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        lisa_data_dir = '/root/paddlejob/workspace/log/code/total_data/dataset'
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir_coco_2017 = ""
        self.image_size = image_size
        self.tokenizer = tokenizer

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        lisa_data_dir,
                        tokenizer,
                        data_args,  # 给一个预处理函数
                    )
                )
            elif dataset == "poster":
                self.all_datasets.append(
                    LazySupervisedDataset(
                        data_path,
                        tokenizer,
                        data_args,
                    )

                )
        print(f"{dataset} is loaded.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):

        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        # if idx % 50:
        #     print(data)
        # inference = False
        return data[0]

