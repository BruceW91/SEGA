import transformers
from llava.mm_utils import tokenizer_image_token
from scripts.bos.bos_client import get_url
from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from PIL import Image
import PIL.Image
import requests
from io import BytesIO
from torch.utils.data import Dataset
import json
import copy
import random
import numpy as np
import torch
PIL.Image.MAX_IMAGE_PIXELS = 933120000
local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def download_image(image_url):
    import logging

    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        logging.error(f"下载失败，HTTP响应状态码:{response.status_code}")
        return None

def rgba2rgb(png):
    png = png.convert('RGBA')
    background = Image.new('RGBA', png.size, (255, 255, 255))

    alpha_composite = Image.alpha_composite(background, png).convert("RGB")
    return alpha_composite

def permute_value(js, key):
    pd = json.loads(js[key][1]['value'])
    random.shuffle(pd)
    js2 = copy.deepcopy(js)
    js2 [key][1]['value'] = json.dumps(pd)
    return js2

cate_dict = {x: i for i, x in enumerate(('Title', 'Subtitle', 'Bodytext', 'Date', 'Name', 'Website', 'Phone number'))}
def give_order(js, key):
    pd = json.loads(js[key][1]['value'])
    pd = sorted(pd, key=lambda x:cate_dict[x['category']])
    js2 = copy.deepcopy(js)
    js2 [key][1]['value'] = json.dumps(pd)
    if key == 'pred':
        inp = js2 [key][0]['value'].split('\ninput: ')[-1]
        inp = [{"category": x["category"], "char_num":x["char_num"]} for x in pd]
        js2 [key][0]['value'] = js2 [key][0]['value'].split('\ninput: ')[0] + '\ninput: ' + json.dumps(inp)
    return js2

def load_remote_json(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()
            return json_data
        else:
            print("Failed to fetch data. Status code:", response.status_code)
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None

def preprocess_multimodal(
    sources,
    data_args
):  
    '''
    image token和内容之间仅用\n分隔
    '''
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_poster(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
):
    # conv = conversation_lib.conv_llava_poster_numch.copy()
    conv = conversation_lib.default_conversation.copy()
    #conv = conversation_lib.conv_llava_poster.copy()    #! 改bug
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        try:
            list_data_dict = json.load(open(data_path, "r"))
        except:
            url = get_url(data_path)
            list_data_dict = load_remote_json(url)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i):
        while True:
            try:
                sources = self.list_data_dict[i]
                
                sources = give_order(sources, 'ocr')
                if isinstance(i, int):
                    sources = [sources]
                assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

                if 'image' in sources[0]:
                    sources = preprocess_multimodal(
                        copy.deepcopy([e["ocr"] for e in sources]),
                        self.data_args)
                    image_file = self.list_data_dict[i]['image_raw']
                    processor = self.data_args.image_processor
                    image = download_image(image_file)
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
                else:
                    sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess_poster(
                    sources,
                    self.tokenizer,
                    has_image=('image' in self.list_data_dict[i]))
                if isinstance(i, int):
                    data_dict = dict(input_ids=data_dict["input_ids"][0],
                                    labels=data_dict["labels"][0])

                # image exist in the data
                if 'image' in self.list_data_dict[i]:
                    data_dict['image'] = image
                elif self.data_args.is_multimodal:
                    # image does not exist in the data, but the model is multimodal
                    crop_size = self.data_args.image_processor.crop_size
                    data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
                return data_dict
            except Exception as e:
                print(e)
                i = np.random.randint(len(self.list_data_dict))
