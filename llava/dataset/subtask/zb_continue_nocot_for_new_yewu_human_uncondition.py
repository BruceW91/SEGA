import json
from PIL import Image
import numpy as np
from .tools import convert_dct_list_text_buttons_for_yewu, give_order, get_bbox_tokens, total_valid, add_newline,convert_dct_list_text_buttons_for_yewu_human
from .unionfind import CustomUnionFind, do_overlap
from ..utils.download import download_image,download_image_for_yewu,download_image_for_yewu_human
from ..utils.imgproc import draw_all,draw_all_for_yewu



def simple_valid(x, sz):
    x0,y0,x1,y1 = bbox = x['Bounding Box']
    ww, hh = sz
    h, w = y1-y0, x1-x0
    if (w < ww and h < hh) and (x0 >=0 and x1<=ww and y0>=0 and y1<=hh) and (w*h>0):
        return True
    else:
        return False
def give_order_ocr(pd):
    pd = sorted(pd, key=lambda x:x['bbox'][:2][::-1])
    return pd
def convert(pk, flip):
    bg = pk[0]
    bgimg = download_image_for_yewu_human(bg)
    # bgimg = download_image_for_yewu(bg)
    ratio = np.random.rand()
    layers, _ = convert_dct_list_text_buttons_for_yewu_human(pk,bgimg.size)
    gua = []
    but = []

    layers = [x for x in layers if simple_valid(x, bgimg.size)]
    layers = give_order(layers)
    num = len(layers)

    ratio = 0
    done = int(np.round(num*ratio))
    if done == num:
        done = num - 1

    done = 0
    bts = []
    # bts = add_newline(json.dumps(gua))
    inpimg = draw_all_for_yewu(bgimg, layers[:done], flip)
    # _ = draw_all(bgimg, layers[done:], flip)
    [ x.pop("Bounding Box") for x in layers]
    # iper = add_newline(json.dumps([{"category": x["category"], "char_num":x["char_num"]} for x in layers[done:]]))  # 这里要改
    temp_list = []
    for x in layers[done:]:
        cur = {}
        cur['category'] = x["category"]

        temp_list.append(cur)
            
    iper = add_newline(json.dumps(temp_list)) 
    iper = []

    dder = [{'bbox':x['bbox']} for x in layers[:done]]
    dder = give_order_ocr(dder)
    dder = [x['bbox'] for x in dder]
    dd = add_newline(json.dumps(dder))
    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a half-finished poster image and a series of text to be added to the poster subsequently, predict the metadata for each text metadata listed below.''' + \
        f'''input: {iper}'''},
        {
            'from': 'gpt',
            'value': f'previous: {dd}\nunderlay: {bts}\n' + add_newline(json.dumps(layers[done:]))
        }
    ]
    return dialog, inpimg