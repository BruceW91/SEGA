# import json
from PIL import Image
import numpy as np
from .unionfind import CustomUnionFind, do_overlap
from .tools import convert_dct_list, give_order, total_valid, add_newline, intersection_area
from .crello import *

import copy
import simplejson as json

from llava.dataset.paint_bbox_utils.paint_for_pku import visualize_output
import os

from llava.dataset.zb_refine_noise_utils.zb_add_noise_for_cgl import add_noise_and_get_score
# from llava.dataset.zb_refine_noise_utils.zb_new_add_noise import add_noise_and_get_score


def process_output(output,size):
    # 把相对左边变成绝对坐标   image.size
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

def give_order_udl(pd):
    pd = sorted(pd, key=lambda x:x[:2][::-1])
    return pd

# 对于加噪版本 temp_pred 是加噪声后自己给出来的  但我开头这里还是要穿进来 方便统一格式

def get_cot_prompt_for_new(layers, temp_saliency_map, cur_img):
    total_prompt = []
    origin_layers = layers.copy()

    cur_layouts, scores = add_noise_and_get_score(layers, temp_saliency_map, cur_img)

    scores_line = {'overlay':0.008,'utilization':0.12,'underlay_effectiveness_strict':0.9,'occlusion':0.2}

    for key,score in scores_line.items():
        cur_line = score
        score_value = scores[key][0]
        if key == 'overlay' and cur_line < score_value:
            total_prompt.append("Text elements are overlayed.")
        if key == 'occlusion' and cur_line < score_value:
            total_prompt.append("Text elements are obscuring the important details in the background image.")
        if key == 'utilization' and cur_line > score_value:
            total_prompt.append("Place the text elements in the blank space.")
        if key == 'underlay_effectiveness_strict' and cur_line > score_value:
            total_prompt.append("Ensure that each underlay on the background image contains a text element.")

    if len(total_prompt) == 0:
        
        total_prompt.append("The layout is fine.")
    return cur_layouts ,total_prompt


def convert(content,img_folder_p,temp_saliency_map):
    # map_dict = {1:'text',2:'logo',3:'underlay'}
    map_dict = {1:'Logo',2:'Text',3:'underlay',4:'embellishment'}

    key,value_list = content
    filename = key.replace(".jpg",".png")
    layers = [ {"category":map_dict[item['category_id']],"absolute_bbox":item['bbox']}  for item in value_list ]
    img_p = os.path.join(img_folder_p, filename)
    cur_img = Image.open(img_p)
    # 要改 layers 和 name

    cur_pred ,prompt_list = get_cot_prompt_for_new(copy.deepcopy(layers), temp_saliency_map, cur_img)

    # 下面获得改动后的绝对坐标 再变成相对的！！！  

    for item in layers:
        item['bbox'] = json.loads(get_bbox_tokens(item['absolute_bbox'], cur_img.size))

    for item in cur_pred:
        item['bbox'] = json.loads(get_bbox_tokens(item['absolute_bbox'], cur_img.size))

    #  以及需要一个类别到文本的转换  下面需要把列表转换成问题
    iper = json.dumps([{"category": x["category"]} for x in layers])
    output = json.dumps([{"category": x["category"], "bbox": x["bbox"]} for x in layers])
    output_pred = json.dumps([{"category": x["category"], "bbox": x["bbox"]} for x in cur_pred])

    layer_for_paint = [ (item['category'],item['absolute_bbox'])  for item in cur_pred  ]

    inpimg = visualize_output(layer_for_paint, cur_img)
    # output_layers = [{"category": x["category"], "bbox": x["bbox"]} for x in layers]
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
# 基于这个没有噪声的版本写