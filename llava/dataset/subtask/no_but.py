import json
from PIL import Image
import numpy as np
from .tools import convert_dct_list_text_buttons, give_order, total_valid, add_newline, get_bbox_tokens
from .unionfind import CustomUnionFind, do_overlap
from ..utils.download import download_image
from ..utils.imgproc import draw_all_button

def give_order_ocr(pd):
    pd = sorted(pd, key=lambda x:x['bbox'][:2][::-1])
    return pd

def simple_valid(x, sz):
    x0,y0,x1,y1 = bbox = x['Bounding Box']
    ww, hh = sz
    h, w = y1-y0, x1-x0
    if (w < ww and h < hh) and (x0 >=0 and x1<=ww and y0>=0 and y1<=hh) and (w*h>0):
        return True
    else:
        return False
def convert(pk, flip):
    bg = pk[0]
    bgimg = download_image(bg)
    ratio = np.random.rand()
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
    num = len(layers)
    done = min(int(num*ratio), num-1)

    inpimg = draw_all_button(bgimg, layers[:done], flip)
    _ = draw_all_button(bgimg, layers[done:], flip)
    [(x.pop('img'), x.pop("Bounding Box")) for x in layers]
    iper = add_newline(json.dumps([{"category": x["category"], "char_num":x["char_num"]} for x in layers[done:]]))
    dialog = [
        {'from': 'human',
   'value': '''<image>\n Please detect all underlays in the image.'''},
        {
            'from': 'gpt',
            'value': add_newline(json.dumps(gua))
        }
    ]
    return dialog, bgimg