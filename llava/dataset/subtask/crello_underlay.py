import json
from PIL import Image
import numpy as np
from .tools import convert_dct_list, give_order, total_valid, add_newline
import random
from .crello import *
def sublist(lst):

    # 生成一个随机的索引列表
    kua = np.random.randint(len(lst))
    indices = random.sample(range(len(lst)), kua)

    # 按照索引列表的顺序从原始列表中选择元素
    sub_lst = [lst[i] for i in sorted(indices)]
    return sub_lst

def give_order_udl(pd):
    pd = sorted(pd, key=lambda x:x[:2][::-1])
    return pd

def give_order_ocr(pd):
    pd = sorted(pd, key=lambda x:x['bbox'][:2][::-1])
    return pd
def convert(s, flip):
    layers = convert_dct_list_crello(s['layers'])
    layers = give_order_ocr(layers)
    layers = sublist(layers)
    bgimg = np.array(get_bg_full(s['layers'][0]['psd_size'], s['bbox'], s['background'], layers))
    inpimg = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers, flip).convert('RGB')
    lys = [x for x in layers if x['Text']!='' and not x.get('bad', False)]
    layers = lys

    udl = s['underlay']
    udl_bbox = []
    w, h = bgimg.shape[1], bgimg.shape[0]
    for x in udl:
        x0,y0,x1,y1 = x
        if not flip:
            udl_bbox.append(json.loads(get_bbox_tokens(x, (w, h))))
        else:
            b2 = [w - x1+1, y0, w - x0-1, y1]
            udl_bbox.append(json.loads(get_bbox_tokens(b2, (w, h))))
    udl_bbox = give_order_udl(udl_bbox)
    # udl_bbox =[{'category':'underlay', 'char_num':0,'bbox':x, 'fontsize':0} for x in udl_bbox]
    judl = json.dumps(udl_bbox)
    [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('orgbox'), x.pop('Angle')) for x in layers]
    iper = add_newline(json.dumps([{"bbox": x["bbox"]} for x in layers]))
    dialog = [
        {'from': 'human',
   'value': '<image>\n Please detect all underlays in the image.'},
        {
            'from': 'gpt',
            'value': judl
        }
    ]
    if flip:
        bgimg = Image.fromarray(np.array(bgimg)[:, ::-1])
    else:
        bgimg = Image.fromarray(np.array(bgimg))
    return dialog, inpimg