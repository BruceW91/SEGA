from torch.utils.data import Dataset
import transformers
import sys
sys.path.append("/home/share/huadjyin/home/wanghaoran/wanghaoran/project/SEGA/LLaVA_poster_multi_task/LLaVA-main")
# from scripts.bos.bos_client import get_url

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

# import 子任务
# from llava.dataset.subtask import no_pred_continue_cot
from llava.dataset.subtask.zb_subtasks import offline_bigposter_no_random_sft_convert_simple

def select_task(pk):
    flip = np.random.rand() > 0.5
    return np.random.choice((offline_bigposter_no_random_sft_convert_simple.convert,))(pk, flip)

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
            with open(data_path, 'rb') as f:
                list_data_dict = pkl.load(f)
        except:
            print("hrer")
            # url = get_url(data_path)
            # response = requests.get(url)
            # list_data_dict = pkl.loads(response.content)
            # list_data_dict = [(a,b,c,d) for a,b,c,d in list_data_dict if len(b) > 0]

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
        for _, sample, _, _ in self.train_dict:
            img_tokens = 128#  if 'image' in sample else 0
            length_list.append(len(sample) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for _, sample, _ , _ in self.train_dict:
            img_tokens = 128#  if 'image' in sample else 0
            length_list.append(len(sample) + img_tokens)
        return length_list

    def __getitem__(self, i):
        while True:
        # try:
            sources = self.train_dict[i]
            dialog, image = select_task(sources)
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

            data_dict['image'] = image
            return data_dict
            # except Exception as e:
            #     # raise
            #     print(e)
            #     i = np.random.randint(len(self.train_dict))


if __name__ == '__main__':
    data_path = '/home/share/huadjyin/home/lishaoshuai/wanghaoran/data/big_poster/0503_raw_offline.pkl'
    # 读取pkl文件
    # with open('/home/share/huadjyin/home/lishaoshuai/wanghaoran/data/big_poster/0503_raw_offline.pkl', 'rb') as f:
    #     list_data_dict = pkl.load(f)
    # # list_data_dict = json.load(open(data_path, "rb"))
    # for item in list_data_dict:
    #     print("jj")
    from PIL import Image
    cu_p = '/home/share/huadjyin/home/lishaoshuai/wanghaoran/data/big_poster/poster_metadata/8515915.psd_5_95f64251c4_bg.png'
    cu_p = '/home/share/huadjyin/home/lishaoshuai/wanghaoran/data/big_poster/poster_metadata/7019951.psd_0_f9d8b474d1_4.png'
    cur_img = Image.open(cu_p)
