from typing import List, Optional
import numpy as np
from PIL import Image
import copy
import cv2
import torch
import re
import json
import traceback
from ..llava_trainer import LLaVATrainer
from ...conversation import conv_templates, SeparatorStyle
from ...mm_utils import tokenizer_image_token, process_images, KeywordsStoppingCriteria



def combine_images(image_list, pad_size=10):
    num_images = len(image_list)
    side_length = int(np.ceil(np.sqrt(num_images)))
    max_dim = max(max(image.shape[0], image.shape[1]) for image in image_list)
    final_image = np.zeros((side_length * (max_dim + 2 * pad_size), side_length * (max_dim + 2 * pad_size), 3), dtype=np.uint8)
    for i, image in enumerate(image_list):
        row = i // side_length
        col = i % side_length
        start_h = row * (max_dim + 2 * pad_size) + pad_size
        start_w = col * (max_dim + 2 * pad_size) + pad_size
        final_image[start_h:start_h+image.shape[0], start_w:start_w+image.shape[1]] = image
    return final_image[:side_length*(max_dim + 2 * pad_size), :side_length*(max_dim + 2 * pad_size)]  # 裁剪成正方形

# 示例用法
image_list = [np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8) for _ in range(9)]  # 生成9个随机图像数组，尺寸为 (100, 200, 3)
final_image = combine_images(image_list)
pil_image = Image.fromarray(final_image)
pil_image.show()


def process_image(img, out_size=720):
    img = img[..., :3]#np.array(rgba2rgb(img))
    size = img.shape[:2]
    image = np.zeros([max(size),max(size),3])
    if size[0] > size[1]:
        diff = (size[0]-size[1]) // 2
        image[:,diff:diff+size[1],:] = img
    else:
        diff = (size[1]-size[0]) // 2
        image[diff:diff+size[0],:,:] = img
    return np.array(cv2.resize(image, (out_size,out_size), interpolation=cv2.INTER_NEAREST),np.int32)


def rgba2rgb(png):
    png = png.convert('RGBA')
    background = Image.new('RGBA', png.size, (255, 255, 255))

    alpha_composite = Image.alpha_composite(background, png).convert("RGB")
    return alpha_composite

def extract_liju_sentences(text, a, b):
    """
    提取来自文本中的 Liju 例句，返回一个列表。
    
    Args:
        text (str): 包含 Liju 例句的文本字符串。
    
    Returns:
        list[str]: 包含所有 Liju 例句的列表，每个例句都是一个字符串。如果没有找到任何例句，则返回一个空列表。
    """
    sentences = []
    # 更新正则表达式以更好地处理双引号
    # pattern = r'"category":\s*"?(.*?)"?\s*(?=\s*"bbox")'

    pattern = a + r':\s*"?(.*?)"?\s*(?=\s*' + b + ')'
    matches = re.findall(pattern, text)
    for sentence in matches:
        # 移除句子两端可能的额外双引号
        clean_sentence = sentence.strip('"')
        if clean_sentence:  # 确保句子内容非空
            zilist = clean_sentence.split(b)
            if len(zilist) > 1:
                clean_sentence = ''.join(zilist[:-1])
            sentences.append(clean_sentence.strip())
    return sentences


def depo(lst, bl):
    num = len(bl)
    if len(lst) < len(bl):
        return lst + ['FAIL'] * (num - len(lst))
    else:
        return lst[:num]
    
import re

def extract_text_in_brackets(text):
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, text)
    return matches
def extract_all(output):
    
    category_raw = extract_liju_sentences(output, '\"category\"', '\"char_num\"')
    category = []
    for x in category_raw:
        try:
            category.append(x.split(',')[0])
        except:
            category.append('FAIL')
    
    bbox_raw = extract_liju_sentences(output, '\"bbox\"', '\"fontsize\"')
    bbox = []
    for x in bbox_raw:
        try:
            x = '[' + extract_text_in_brackets(x)[0] + ']'
            ga = eval(x)
            if (isinstance(ga, list) or isinstance(ga, tuple)) and len(ga) == 4:
                bbox.append(ga)
            else:
                bbox.append('FAIL')
        except:
            bbox.append('FAIL')

    char_num_raw = extract_liju_sentences(output, '\"char_num\"', '\"bbox\"')
    char_num = []
    for x in char_num_raw:
        try:
            char_num.append(json.loads(x.split(',')[0]))
        except:
            char_num.append('FAIL')
    
    fontsize_raw = extract_liju_sentences(output, '\"fontsize\"', '\"fontcolor\"')
    fontsize = []
    for x in fontsize_raw:
        try:
            fontsize.append(json.loads(x.split(',')[0]))
        except:
            fontsize.append('FAIL')


    fontcolor_raw = extract_liju_sentences(output, '\"fontcolor\"', '\"alignment\"')
    fontcolor = []
    for x in fontcolor_raw:
        try:
            fontcolor.append(json.loads(x.split('"')[0]))
        except:
            fontcolor.append('FAIL')

    alignment_raw = extract_liju_sentences(output, '\"alignment\"', '}')
    alignment = []
    for x in alignment_raw:
        try:
            alignment.append(x.split(',')[0])
        except:
            alignment.append('FAIL')
    
    category, char_num, fontsize, fontcolor, alignment = [depo(x, bbox) for x in (category, char_num, fontsize, fontcolor, alignment)]
    others = [category, bbox, char_num, fontsize, fontcolor, alignment]
    inder = [i for i, x in enumerate(bbox) if x != 'FAIL']
    for y in others:
        y = [y[x] for x in inder]

    return list(zip(*others))

def extract_bb(output):
    category_raw = extract_liju_sentences(output, '\"category\"', '\"bbox\"')
    category = []
    for x in category_raw:
        try:
            category.append(x.split(',')[0])
        except:
            category.append('FAIL')
    
    bbox_raw = extract_liju_sentences(output, '\"bbox\"', '}')
    bbox = []
    for x in bbox_raw:
        try:
            x = '[' + extract_text_in_brackets(x)[0] + ']'
            ga = eval(x)
            if (isinstance(ga, list) or isinstance(ga, tuple)) and len(ga) == 4:
                bbox.append(ga)
            else:
                bbox.append('FAIL')
        except:
            bbox.append('FAIL')

    char_num_raw = extract_liju_sentences(output, '\"char_num\"', '\"bbox\"')
    char_num = []
    for x in char_num_raw:
        try:
            char_num.append(json.loads(x.split(',')[0]))
        except:
            char_num.append('FAIL')
    
    fontsize_raw = extract_liju_sentences(output, '\"fontsize\"', '\"fontcolor\"')
    fontsize = []
    for x in fontsize_raw:
        try:
            fontsize.append(json.loads(x.split(',')[0]))
        except:
            fontsize.append('FAIL')


    fontcolor_raw = extract_liju_sentences(output, '\"fontcolor\"', '\"alignment\"')
    fontcolor = []
    for x in fontcolor_raw:
        try:
            fontcolor.append(json.loads(x.split('"')[0]))
        except:
            fontcolor.append('FAIL')

    alignment_raw = extract_liju_sentences(output, '\"alignment\"', '}')
    alignment = []
    for x in alignment_raw:
        try:
            alignment.append(x.split(',')[0])
        except:
            alignment.append('FAIL')
    
    category, char_num, fontsize, fontcolor, alignment = [depo(x, bbox) for x in (category, char_num, fontsize, fontcolor, alignment)]
    others = [category, bbox, char_num, fontsize, fontcolor, alignment]
    inder = [i for i, x in enumerate(bbox) if x != 'FAIL']
    for y in others:
        y = [y[x] for x in inder]

    return list(zip(*others))
def draw_one(npimg, tp, colort):

    label = tp[0][:-1]
    output = ot = process_output(tp[1], npimg.shape[:2][::-1])
    charnum = tp[2]
    fontsize = tp[3]
    color = tp[4]
    al = tp[5]
    text = f"{label}/{charnum}/{fontsize}/{color}/{al}"
    cv2.rectangle(npimg,tuple(ot[:2]),tuple(ot[2:]),colort,3)
    cv2.putText(img=npimg, text = text, org=tuple(output[:2]), \
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=colort,thickness=3)
    return npimg

def visualize_output(output, image, color):
    res = extract_all(output)
    print(res)
    npimg = np.array(rgba2rgb(image))
    for x in res:
        draw_one(npimg, x, color)
    return npimg

def visualize_output_bb(output, image, color):
    res = extract_bb(output)
    print(res)
    npimg = np.array(rgba2rgb(image))
    for x in res:
        draw_one(npimg, x, color)
    return npimg

def extract_liju_sentences(text, a, b):
    import re
    """
    提取来自文本中的 Liju 例句，返回一个列表。
    
    Args:
        text (str): 包含 Liju 例句的文本字符串。
    
    Returns:
        list[str]: 包含所有 Liju 例句的列表，每个例句都是一个字符串。如果没有找到任何例句，则返回一个空列表。
    """
    sentences = []
    # 更新正则表达式以更好地处理双引号
    # pattern = r'"category":\s*"?(.*?)"?\s*(?=\s*"bbox")'

    pattern = a + r':\s*"?(.*?)"?\s*(?=\s*' + b + ')'
    matches = re.findall(pattern, text)
    for sentence in matches:
        # 移除句子两端可能的额外双引号
        clean_sentence = sentence.strip('"')
        if clean_sentence:  # 确保句子内容非空
            zilist = clean_sentence.split(b)
            if len(zilist) > 1:
                clean_sentence = ''.join(zilist[:-1])
            sentences.append(clean_sentence.strip())
    return sentences

def process_output(output,size):
    max_size = max(list(size))
    min_size = min(list(size))
    size_list = list(size)
    diff = (max_size-min_size)//2
    min_id = size_list.index(min_size)

    output = [int(float(i)*max_size) for i in output]
    if min_id == 0: 
        output[0] = output[0]-diff
        output[2] = output[2]-diff
    else:
        output[1] = output[1]-diff
        output[3] = output[3]-diff

    return output
def generate_bbox(image, prompt, conv_mode, model, image_processor, tokenizer):
    # prompt = '''<image>\nText 0: Harmony With Nature.\nText 1: Transforming Your Space, Sustainably.\nText 2: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Quis ipsum suspendisse ultrices gravida. .\nText 3: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Quis ipsum suspendisse ultrices gravida. .\nText 4: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Quis ipsum suspendisse ultrices gravida. .\nWhat should be the bounding box and the typographic attributes(font size, justification, color) of Text 0? Answer the question in the format [fontsize,[x_min, y_min, x_max, y_max],justification,[red,green,blue]].'''
    promptb = prompt.split('<image>\n')[-1].strip()
    prompt = '<image>\n' + promptb
    image = rgba2rgb(image)
    conv = conv_templates[conv_mode].copy()
    conversations = []
    conv.append_message(conv.roles[0], prompt)
    conversations.append(conv.get_prompt() + ' ASSISTANT:')
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0).to(model.device)
    images_tensor = process_images(
        [image, ],
        image_processor,
        model.config
    ).to(model.device, dtype=model.dtype)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        # import pdb; pdb.set_trace()
        output_ids = model.generate(
            inputs=input_ids,
            images=images_tensor,
            do_sample=True,
            temperature=0.2,
            top_p=1.0,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria],
        )
        # output_ids = model.generate(input_ids, images=images_tensor, do_sample=False, temperature=0, top_p=None, num_beams=1, max_new_tokens=512, use_cache=True, stopping_criteria=[stopping_criteria],)
    return output_ids

def evaluate(
        trainer: LLaVATrainer,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        from ...dataset.subtask import ocr_cotv3 as ocr
        from ...dataset.subtask import pred_textnoc_cot
        valset = trainer.train_dataset.valset

        flag = True
        images_ = []
        for x in valset:
            try:
                flag = not flag
                dialog, image = pred_textnoc_cot.convert(x, flag)
                output_ids = generate_bbox(image, dialog[0]['value'], 'poster_llava_general_wjh_0407', trainer.model, trainer.model.get_vision_tower().image_processor, trainer.tokenizer)
                output = trainer.tokenizer.batch_decode(output_ids)[0]
                print(output)
                res = visualize_output(dialog[1]['value'], image, (0,0,255))
                try:
                    res = visualize_output(output, Image.fromarray(res), (255,0,0))
                except:
                    traceback.print_exc()
                images_.append(process_image(res))
            except:
                traceback.print_exc()
        if len(images_) > 0:
            images = combine_images(images_)[None].transpose(0,3,1,2)
            trainer.tb_writer.add_images("Text",np.array(images/255),global_step = trainer.state.global_step)
            trainer.tb_writer.flush()

        flag = True
        images_ = []
        for x in valset:
            try:
                flag = not flag
                dialog, image = ocr.convert(x, flag)
                output_ids = generate_bbox(image, dialog[0]['value'], 'poster_llava_general_wjh_0407', trainer.model, trainer.model.get_vision_tower().image_processor, trainer.tokenizer)
                output = trainer.tokenizer.batch_decode(output_ids)[0]
                print(output)
                res = visualize_output(dialog[1]['value'], image, (0,0,255))
                try:
                    res = visualize_output(output, Image.fromarray(res), (255,0,0))
                except:
                    traceback.print_exc()
                images_.append(process_image(res))
            except:
                traceback.print_exc()
        if len(images_) > 0:
            images = combine_images(images_)[None].transpose(0,3,1,2)
            trainer.tb_writer.add_images("OCR",np.array(images/255),global_step = trainer.state.global_step)
            trainer.tb_writer.flush()


