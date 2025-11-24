import json
from PIL import Image
import numpy as np
from .tools import convert_dct_list_text_buttons_for_0530, give_order, get_bbox_tokens, total_valid, add_newline,convert_dct_list_text_buttons
from .unionfind import CustomUnionFind, do_overlap
from ..utils.download import download_image
from ..utils.imgproc import draw_all

import random

def simple_valid(x, sz):
    x0,y0,x1,y1 = bbox = x['Bounding Box']
    ww, hh = sz
    h, w = y1-y0, x1-x0
    if (w < ww and h < hh) and (x0 >=0 and x1<=ww and y0>=0 and y1<=hh) and (w*h>0):
        return True
    else:
        return False

def zb_perturb_bbox(bbox, translate_ratio=0.3, scale_ratio=0.2):
    """
    对0-1归一化的bbox进行平移和缩放的扰动。
    
    参数:
    bbox: list or tuple, 形如[x_min, y_min, x_max, y_max]的归一化bbox，其中每个值都在[0, 1]之间。
    translate_ratio: float, 平移比例，取值范围建议在0到1之间，表示bbox可以在其原始位置基础上最大平移自身宽度或高度的translate_ratio倍。
    scale_ratio: float, 缩放比例，可以是正数也可以是负数，表示bbox可以在其原始尺寸基础上最大放大或缩小scale_ratio倍。
    
    返回:
    list, 扰动后的bbox。
    """
    # 解构bbox
    x_min, y_min, x_max, y_max = bbox
    
    # 计算bbox的宽度和高度
    width = x_max - x_min
    height = y_max - y_min
    
    # 计算平移量
    translate_x = (x_max - x_min) * translate_ratio
    translate_y = (y_max - y_min) * translate_ratio
    
    # 随机决定平移的方向（正向或负向）
    random_sign_x = -1 if random.random() < 0.5 else 1
    random_sign_y = -1 if random.random() < 0.5 else 1
    dx = translate_x * random_sign_x
    dy = translate_y * random_sign_y
    
    # 应用平移
    new_x_min = max(0, x_min + dx)
    new_y_min = max(0, y_min + dy)
    new_x_max = min(1, x_max + dx)
    new_y_max = min(1, y_max + dy)
    
    # 计算缩放因子
    scale_factor = 1 + scale_ratio * (2 * random.random() - 1)  # 保证了scale_factor在[1-scale_ratio, 1+scale_ratio]之间
    
    # 应用缩放
    new_width = width * scale_factor
    new_height = height * scale_factor
    
    # 确保缩放后bbox仍在[0, 1]范围内
    new_x_max = min(1, new_x_min + new_width)
    new_y_max = min(1, new_y_min + new_height)
    
    # return [new_x_min, new_y_min, new_x_max, new_y_max]
    return [round(val, 3) for val in [new_x_min, new_y_min, new_x_max, new_y_max]]
    
def scale_bbox(bbox, ratio, size):
    x0, y0, x1, y1 = bbox
    w, h = size
    width = x1 - x0
    height = y1 - y0
    
    new_width = width * ratio
    new_height = height * ratio
    
    x0_new = x0 - (new_width - width) / 2
    y0_new = y0 - (new_height - height) / 2
    x1_new = x0_new + new_width
    y1_new = y0_new + new_height

    # Check if the new bbox is out of the given range
    if x0_new < 0 or y0_new < 0 or x1_new > w or y1_new > h:
        # Adjust the ratio
        ratio_x = (w - x1) / width
        ratio_y = (h - y1) / height
        ratio = (min((w - x1) / w, (h - y1) / h, x0 / w, y0 / h)*2)
        ratio += 1
        # Recalculate the new bbox with the adjusted ratio
        new_width = width * ratio
        new_height = height * ratio
        
        x0_new = x0 - (new_width - width) / 2
        y0_new = y0 - (new_height - height) / 2
        x1_new = x0_new + new_width
        y1_new = y0_new + new_height
    
    return [int(x) for x in (x0_new, y0_new, x1_new, y1_new)], ratio


def translate_bbox(bbox, sizes):
    def _func(bbox, ratio):
        return (x+ratio for x in bbox)
    r = np.array(sizes).max()
    x0, y0, x1, y1 = bbox
    ratio = np.random.randn()/6
    if ratio > 0:
        ratio = min(sizes[0]-x1, ratio * r)
    else:
        ratio = max(-x0, ratio * r)
    x0, x1 = _func([x0, x1], ratio)

    ratio = np.random.randn()/6
    if ratio > 0:
        ratio = min(sizes[1]-y1, ratio * r)
    else:
        ratio = max(-y0, ratio * r)
    y0, y1 = _func([y0, y1], ratio)
    return [int(x) for x in (x0, y0, x1, y1)]

def perturb_bbox(bbox, sizes):
    bbox = translate_bbox(bbox, sizes)
    ratio = np.random.randn()
    if ratio > 0:
        ratio += 1
    else:
        ratio = 1/(1+abs(ratio))
    rebbox, ratio = scale_bbox(bbox, ratio, sizes)
    w, h = rebbox[2] - rebbox[0], rebbox[3] - rebbox[1]
    if min(w, h) < 5:
        ratio = 5 / min(w, h)
        rebbox, ratio = scale_bbox(bbox, ratio, sizes)
    
    return rebbox, ratio    


def give_order_ocr(pd):
    pd = sorted(pd, key=lambda x:x['bbox'][:2][::-1])
    return pd
def convert(pk, flip, in_layer=-1):
    '''
    我这个就是专用的dpo convert   全预测任务   输出加 与 不加扰动的dialog
    '''
    bg = pk[0]
    bgimg = download_image(bg)
    layer_ratio = np.random.rand()
    layers, _ = convert_dct_list_text_buttons(pk)
    gua = []
    w, h = bgimg.size
    for but in pk[-1]:
    # x0, y0, x1, y1 = [int(_) for _ in but]
        if not flip:
            gua.append(json.loads(get_bbox_tokens([int(_) for _ in but], (w, h))))
        else:
            x0, y0, x1, y1 = but
            gua.append(json.loads(get_bbox_tokens([w - x1, y0, w - x0, y1], (w, h))))

    layers = [x for x in layers if simple_valid(x, bgimg.size)]
    layers = give_order(layers)
    origlayers = layers.copy()
    num = len(layers)
    # 到这相当于是一个原生的完整版layers

    #! 加入bbox位置扰动  
    orglayers = []
    for x in layers:
        y = {}

        for k, v in x.items():
            y[k] = v
        if True:  # 永久扰动
            # y['modified'] = True
            y['Bounding Box'], ratio = perturb_bbox(x['Bounding Box'], bgimg.size)
            y["fontsize"] = np.around(x["fontsize"] * ratio, 2)
            y['bbox'] = json.loads(get_bbox_tokens(y['Bounding Box'], bgimg.size))
            # float_list = json.loads(x['fontcolor'])
            # y['fontcolor'] = [np.around(1 - float(x),3) for x in float_list]

        orglayers.append(y)     # orglayers保存的视频原始信息

    # 设定层数 
    if in_layer != -1:
        done = in_layer
    else:
        done = int(np.round(num*layer_ratio))
        if done == num:
            done = done - 1
    
    # print("num,done:",num,done)
    perturb_gua = [ zb_perturb_bbox(item) for item in gua.copy() ]
    perturb_bts = add_newline(json.dumps(perturb_gua))
    # bts = []
    bts = add_newline(json.dumps(gua))
    inpimg = draw_all(bgimg, layers[:done], flip)
    _ = draw_all(bgimg, layers[done:], flip)

    [(x.pop('thought'), x.pop('buttons'), x.pop('Text'), x.pop('img'), x.pop("Bounding Box")) for x in origlayers]
    [(x.pop('thought'), x.pop('buttons'), x.pop('Text'), x.pop('img'), x.pop("Bounding Box")) for x in orglayers]
    
    iper = add_newline(json.dumps([{"category": x["category"], "char_num":x["char_num"]} for x in origlayers[done:]]))
    dder = [{'bbox':x['bbox']} for x in origlayers[:done]]
    dder = give_order_ocr(dder)
    dder = [x['bbox'] for x in dder]
    dd = add_newline(json.dumps(dder))
    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a half-finished poster image and a series of text to be added to the poster subsequently, predict the metadata for each text metadata listed below.''' + \
        f'''input: {iper}'''},
        {
            'from': 'gpt',
            'value': f'previous: {dd}\nunderlay: {bts}\n' + add_newline(json.dumps(origlayers[done:]))
        }
    ]

    nega_dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a half-finished poster image and a series of text to be added to the poster subsequently, predict the metadata for each text metadata listed below.''' + \
        f'''input: {iper}'''},
        {
            'from': 'gpt',
            'value': f'previous: {dd}\nunderlay: {perturb_bts}\n' + add_newline(json.dumps(orglayers[done:]))
        }
    ]

    # dialog  我要给出去两个版本
    return dialog, nega_dialog, inpimg, done