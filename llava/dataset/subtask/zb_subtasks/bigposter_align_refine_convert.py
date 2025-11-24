import json
from PIL import Image
import numpy as np
from llava.dataset.subtask.unionfind import CustomUnionFind, do_overlap
from llava.dataset.subtask.tools import convert_dct_list, give_order, total_valid, add_newline, intersection_area
from llava.dataset.subtask.crello import *

from ..tools import convert_dct_list_text_buttons

from llava.dataset.utils.imgproc import draw_all,draw_all_zb_re

from copy import deepcopy

import os

def rgba2rgb(png):
    png = png.convert('RGBA')
    background = Image.new('RGBA', png.size, (255, 255, 255))

    alpha_composite = Image.alpha_composite(background, png).convert("RGB")
    return alpha_composite

def convert_dct_list_crello_simple(layers):
    # bg, layers, *_ = pk
    psdsize = layers[0]['psd_size']
    # bgimg = download_imc reage(bg)
    lys = []
    for x in layers:
        dct = {}
        dct['Angle'] = x['Angle']
        dct['Text'] = x['Text']
        dct['orgbox'] = x['orgbox']
        dct['category'] = x['label']
        dct['char_num'] = len(x['Text'].strip())
        dct['bbox'] = json.loads(get_bbox_tokens_2(x['Bounding Box'], psdsize))
        dct['Bounding Box'] = x['Bounding Box']
        dct['fontsize'] = normalize_fontsize(x)
        dct['fontcolor'] = get_color(x['FillColor'])
        dct['img'] = x['img']

        lys.append(dct)
    return lys

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
    # thresholds = [0.001, 0.2, 0.07, 0.52, 0.027, 0.009, 0.95, 0.95, 0.98]
    # thresholds = [0.001, 0.2, 0.14, 0.35, 0.027, 0.006, 0.95, 0.95]
    thresholds = [0.001, 0.2, 0.14, 0.25, 0.03, 0.01, 0.8, 0.8]
    if scores[0] > thresholds[0]:
        # total_prompt.append("Text elements require stricter alignment")
        pass
    if scores[5] > thresholds[5]:#越低越好
        total_prompt.append("Text elements are overlayed")
    # if scores[2] < thresholds[2]:  #utilization
    #     total_prompt.append("Place the text elements in the blank space.")
    if scores[4] > thresholds[4]:#越低越好  
        total_prompt.append("Text elements should not be placed on backgrounds with drastic color changes.")
        # pass
    if scores[3] > thresholds[3]:  #越低越好
        total_prompt.append("Text elements are obscuring the important details in the background image.")
    if scores[7] < thresholds[7]:
        total_prompt.append("Ensure that each underlay on the background image contains a text element.")
    # if scores[8] < thresholds[8]:
    #     total_prompt.append("There should not be elements with overly small areas.")
    if len(total_prompt) == 0:
        total_prompt.append("The layout is almost fine.")
    return total_prompt


def convert(content, flip, scores, temp_pred):
    # 输入有两个信息流 一个原始的 gt 一个当前的

    bg_img_c = temp_pred['image']
    bgimg = rgba2rgb(deserail(bg_img_c))
    psdsize = bgimg.size

    pred_layers = temp_pred['layers']
    need_list = [ {item['label']:item['Bounding Box']} for item in pred_layers if item['label']!='underlay' ]
    for item in need_list:
        for k,v in item.items():
            item[k] = json.loads(get_bbox_tokens(v, psdsize))

    layers,_ = convert_dct_list_text_buttons(content)
    layers = give_order(layers)

    lys = [x for x in layers if x['Text']!='' and not x.get('bad', False)]
    layers = lys
    
    w,h = psdsize
    udl = content[-1]
    udl_bbox = []
    for x in udl:
        x0,y0,x1,y1 = x
        if not flip:
            udl_bbox.append(json.loads(get_bbox_tokens(x, psdsize)))
        else:
            b2 = [w - x1+1, y0, w - x0-1, y1]
            udl_bbox.append(json.loads(get_bbox_tokens(b2, psdsize)))
    udl_bbox = give_order_udl(udl_bbox)
    # udl_bbox =[{'category':'underlay', 'char_num':0,'bbox':x, 'fontsize':0} for x in udl_bbox]
    judl = json.dumps(udl_bbox)

    # 之前这边的图 没画进去
    # np_img = np.array(bgimg)
    pred_layers_for_draw = [x for x in pred_layers if x.get('Text', False)]
    inpimg = draw_all_zb_re(bgimg,pred_layers_for_draw).convert('RGB')


    # 其实也可以改成在这里多pop 
    [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('buttons'), x.pop('thought'),x.pop('fontcolor'),x.pop('fontsize'),x.pop('alignment'),x.pop('char_num')) for x in layers]

    # iper = add_newline(json.dumps([{"category": x["category"], "char_num":x["char_num"]} for x in layers]))

    dd = [x['bbox'] for x in layers]
    dd = give_order_udl(dd)
    dd = json.dumps(dd) 

    prompt_list = get_cot_prompt_for_align(scores)
    total_str = ""
    for i in range(len(prompt_list)):
        total_str += f"{i}. {prompt_list[i]} "
    reasons = "Reasons:" + total_str

    # need_list = [ for label,_ in need_list ]
    need_str = json.dumps(need_list)

    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a poster image and a poster laypout, please read the underlay, refine the layouts and give reasons.\n''' + \
        f'''input: Layouts to be refined:" {need_str}'''},
        {
            'from': 'gpt',
            'value': f'underlay: {judl}\n'+ f'{reasons}\n'+add_newline(json.dumps(layers)).replace('\n','')
        }
    ]
    return dialog, inpimg

def convert_no_score(content, flip, temp_pred):
    # 输入有两个信息流 一个原始的 gt 一个当前的
    flip = False
    bg_img_c = temp_pred['image']
    bgimg = rgba2rgb(deserail(bg_img_c))
    bbb_img = rgba2rgb(deserail(bg_img_c))
    psdsize = bgimg.size

    pred_layers = temp_pred['layers']
    need_list = [ {item['label']:item['Bounding Box']} for item in pred_layers if item['label']!='underlay' ]
    for item in need_list:
        for k,v in item.items():
            item[k] = json.loads(get_bbox_tokens(v, psdsize))

    layers,_ = convert_dct_list_text_buttons(content)

    origin_lys = deepcopy(layers)

    layers = give_order(layers)

    lys = [x for x in layers if x['Text']!='' and not x.get('bad', False)]
    layers = lys
    
    w,h = psdsize
    udl = content[-1]
    udl_bbox = []
    for x in udl:
        x0,y0,x1,y1 = x
        if not flip:
            udl_bbox.append(json.loads(get_bbox_tokens(x, psdsize)))
        else:
            b2 = [w - x1+1, y0, w - x0-1, y1]
            udl_bbox.append(json.loads(get_bbox_tokens(b2, psdsize)))
    udl_bbox = give_order_udl(udl_bbox)

    judl = json.dumps(udl_bbox)


    pred_layers_for_draw = [x for x in pred_layers if x.get('Text', False)]
    inpimg = draw_all_zb_re(bgimg.copy(),pred_layers_for_draw).convert('RGB')


    # 其实也可以改成在这里多pop 
    [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('buttons'), x.pop('thought'),x.pop('fontcolor'),x.pop('fontsize'),x.pop('alignment')) for x in layers]

    dd = [x['bbox'] for x in layers]
    dd = give_order_udl(dd)
    dd = json.dumps(dd) 


    reasons = "Reasons:" 

    # need_list = [ for label,_ in need_list ]
    need_str = json.dumps(need_list)

    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a poster image and a poster laypout, please read the underlay, refine the layouts and give reasons.\n''' + \
        f'''input: Layouts to be refined:" {need_str}'''},
        {
            'from': 'gpt',
            'value': f'underlay: {judl}\n'+ f'{reasons}\n'+add_newline(json.dumps(layers))
        }
    ]
    return dialog, inpimg, origin_lys, bbb_img




def conver_for_gpt(s, temp_pred):
    
    pred_layers = temp_pred['layers']
    need_list = [ {item['label']:item['Bounding Box']} for item in pred_layers ]
    for item in need_list:
        for k,v in item.items():
            item[k] = json.loads(get_bbox_tokens_2(v, s['layers'][0]['psd_size']))
    # 我真吐了  这边还要改回相对坐标 查看是不是对齐了 ！！！
    # 分析之后 这个是正经的在没pad的图像上的绝对坐标

    need_str = json.dumps(need_list)

    return need_str
    # return dialog, inpimg