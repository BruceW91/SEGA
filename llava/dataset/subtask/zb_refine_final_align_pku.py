import json
from PIL import Image
import numpy as np
from .unionfind import CustomUnionFind, do_overlap
from .tools import convert_dct_list, give_order, total_valid, add_newline, intersection_area
from .crello import *

import os
import copy

from llava.dataset.paint_bbox_utils.paint_for_pku import visualize_output


def give_order_udl(pd):
    pd = sorted(pd, key=lambda x:x[:2][::-1])
    return pd

def get_cot_prompt(scores):
    total_prompt = []
    thresholds = [0.001, 0.2, 0.11, 0.35, 0.9, 0.008, 0.95, 0.95]
    if scores[0] > thresholds[0]:
        total_prompt.append("Text elements require stricter alignment")
    if scores[1] > thresholds[1]:
        total_prompt.append("Text elements are overlayed")
    if scores[2] < thresholds[2]:
        total_prompt.append("Place the text elements in the blank space.")
    if scores[3] > thresholds[3]:
        total_prompt.append("Text elements are obscuring the important details in the background image.")
    if scores[7] < thresholds[7]:
        total_prompt.append("Ensure that each underlay on the background image contains a text element.")
    if len(total_prompt) == 0:
        total_prompt.append("The layout is fine.")
    return total_prompt

def get_cot_prompt_for_align(scores):
    total_prompt = []
    thresholds = [0.001, 0.3, 0.2, 0.13, 0.9, 0.008, 0.95, 0.9]
    if scores[0] > thresholds[0]:
        # total_prompt.append("Text elements require stricter alignment")
        pass
    if scores[1] > thresholds[1]:
        total_prompt.append("Text elements are overlayed")
    if scores[2] < thresholds[2]:
        total_prompt.append("Place the text elements in the blank space.")
        # pass
    if scores[3] > thresholds[3]:
        total_prompt.append("Text elements are obscuring the important details in the background image.")
    if scores[7] < thresholds[7]:
        total_prompt.append("Ensure that each underlay on the background image contains a text element.")
    if len(total_prompt) == 0:
        total_prompt.append("The layout is fine.")
    return total_prompt


def convert(content, scores, temp_pred, img_folder_p):
    map_dict = {1:'text',2:'logo',3:'underlay'}
    key,value_list = content
    img_p = os.path.join(img_folder_p, key)
    cur_img = Image.open(img_p)

    pred_layers = temp_pred['layers']
    need_list = [ {item['label']:item['Bounding Box']} for item in pred_layers ]
    visual_list = [ [item['label'], item['Bounding Box']] for item in pred_layers ]
    # origin_need_list = copy.deepcopy(need_list)
    for item in need_list:
        for k,v in item.items():
            item[k] = json.loads(get_bbox_tokens(v, cur_img.size))

    
    layers = [ {"category":map_dict[category],"absolute_bbox":bbox}  for category,bbox in value_list ]
    img_p = os.path.join(img_folder_p, key)
    cur_img = Image.open(img_p)

    prompt_list = get_cot_prompt_for_align(scores)

    # 下面获得改动后的绝对坐标 再变成相对的！！！  

    for item in layers:
        item['bbox'] = json.loads(get_bbox_tokens(item['absolute_bbox'], cur_img.size))
    for item in visual_list:
        item.append(get_bbox_tokens(item[1], cur_img.size))


    #  以及需要一个类别到文本的转换  下面需要把列表转换成问题
    iper = json.dumps([{"category": x["category"]} for x in layers])
    output = json.dumps([{"category": x["category"], "bbox": x["bbox"]} for x in layers])
    
    output_pred = json.dumps([{"category": x[0], "bbox": x[2]} for x in visual_list])

    # 我要画 temp 不能用 value_list   要用绝对坐标
    inpimg = visualize_output(visual_list, cur_img)

    father_folder = 'infer_out_refine/debug_look'
    # orig_f = os.path.join(father_folder, 'orig_img_noise')
    cur_f  = os.path.join(father_folder, 'cur_img_noise')
    # os.makedirs(orig_f, exist_ok=True)
    os.makedirs(cur_f, exist_ok=True)

    # in_orig_img.save(os.path.join(orig_f, s['layers'][0]['id'] + ".png"))
    inpimg.save(os.path.join(cur_f, key))

    total_str = ""
    for i in range(len(prompt_list)):
        total_str += f"{i}. {prompt_list[i]} "
    reasons = "Reasons:" + total_str


    dialog = [
        {'from': 'human',
    'value': '''<image>\n Given a poster image and a poster laypout, please refine the layouts and give reasons.\n''' + \
        f'''input: Layouts to be refined:" {output_pred}'''},
        {
            'from': 'gpt',
            'value': f'{reasons}\n'+f'''{output}'''
        }
    ]
    return dialog, inpimg


def convert_no_score(content, temp_pred, img_folder_p):
    map_dict = {1:'text',2:'logo',3:'underlay'}
    key,value_list = content
    img_p = os.path.join(img_folder_p, key)
    cur_img = Image.open(img_p)

    pred_layers = temp_pred['layers']
    need_list = [ {item['label']:item['Bounding Box']} for item in pred_layers ]
    visual_list = [ [item['label'], item['Bounding Box']] for item in pred_layers ]
    # origin_need_list = copy.deepcopy(need_list)
    for item in need_list:
        for k,v in item.items():
            item[k] = json.loads(get_bbox_tokens(v, cur_img.size))

    
    layers = [ {"category":map_dict[category],"absolute_bbox":bbox}  for category,bbox in value_list ]
    img_p = os.path.join(img_folder_p, key)
    cur_img = Image.open(img_p)


    for item in layers:
        item['bbox'] = json.loads(get_bbox_tokens(item['absolute_bbox'], cur_img.size))
    for item in visual_list:
        item.append(get_bbox_tokens(item[1], cur_img.size))


    #  以及需要一个类别到文本的转换  下面需要把列表转换成问题
    iper = json.dumps([{"category": x["category"]} for x in layers])
    output = json.dumps([{"category": x["category"], "bbox": x["bbox"]} for x in layers])
    
    # output_pred = json.dumps(need_list)
    output_pred = json.dumps([{"category": x[0], "bbox": x[2]} for x in visual_list])
    # output_pred = json.dumps([{"category": x[0], "bbox": x[1]} for x in visual_list])

    # 我要画 temp 不能用 value_list   要用绝对坐标
    inpimg = visualize_output(visual_list, cur_img)


    dialog = [
        {'from': 'human',
    'value': '''<image>\n Given a poster image and a poster laypout, please refine the layouts and give reasons.\n''' + \
        f'''input: Layouts to be refined:" {output_pred}'''},
        {
            'from': 'gpt',
            'value': f'''{output}'''
        }
    ]
    return dialog, inpimg, cur_img