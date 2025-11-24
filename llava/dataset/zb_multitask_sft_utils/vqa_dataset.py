import json
import os
import random
import copy

import cv2
import torch
from PIL import Image

import torch.nn.functional as F
from transformers import CLIPImageProcessor

from llava.dataset.preproc import preprocess_multimodal, preprocess_poster
from llava.dataset.utils.imgproc import  rgba2rgb

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

class VQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        data_args,
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        vqa_data="llava_instruct_150k",
    ):
        self.exclude_val = exclude_val
        self.num_classes_per_sample = num_classes_per_sample
        self.data_args = data_args

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        # self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = data_args.image_processor

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        print("vqa_data: ", len(self.vqa_data))

    def __len__(self):
        return self.vqa_data

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        # 用pil读图

        image = Image.open(image_path).convert('RGB')
        image = rgba2rgb(image)

        sources = preprocess_multimodal(
        [copy.deepcopy(item['conversations']),],
        self.data_args)


        image = expand2square(image, tuple(int(x*255) for x in self.clip_image_processor.image_mean))
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][
            0
        ]  # preprocess image for clip

        data_dict = preprocess_poster(
            sources,
            self.tokenizer,
            has_image=True)

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0])
        data_dict["image"] = image_clip
        # if idx % 50:
            # print('data:', sources)
        
        return data_dict
