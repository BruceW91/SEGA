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
from pathlib import Path

# import 子任务
from .subtask import zb_uncondition_sft_convert.py

def select_task(pk):
    s = pkl.load(open(pk, 'rb'))
    flip = np.random.rand() > 0.5
    return np.random.choice((zb_uncondition_sft_convert.convert,))(s, flip)

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
            list_data_dict = sorted(list(Path(data_path).glob('*.pkl')))

            # list_data_dict = json.load(open(data_path, "r"))
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
                dialog, image, layers_num = select_task(sources)
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

