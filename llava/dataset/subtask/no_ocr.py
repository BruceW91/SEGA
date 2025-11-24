import json
from .tools import convert_dct_list, give_order, total_valid, add_newline
from .unionfind import CustomUnionFind, do_overlap
from ..utils.download import download_image
from ..utils.imgproc import draw_all

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
    bgimg = download_image(bg)
    try:
        layers = convert_dct_list(pk)
    except Exception as e:
        for _ in range(20):
            print('pk:', str(pk))
        raise e
    
    layers = [x for x in layers if simple_valid(x, bgimg.size)]
    layers = give_order_ocr(layers)

    inpimg = draw_all(bgimg, layers, flip)
    [(x.pop('img'), x.pop("Bounding Box")) for x in layers]
    dialog = [
        {'from': 'human',
   'value': '<image>\n Please detect all text blocks in the image, and output the corresponding category, char_num, bbox, fontsize, fontcolor, alignment for each text block.'},
        {
            'from': 'gpt',
            'value': f'There are total {len(layers)} text blocks in the image.\n' + add_newline(json.dumps(layers))
        }
    ]
    return dialog, inpimg
