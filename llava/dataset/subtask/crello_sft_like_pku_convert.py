import json
from PIL import Image
import numpy as np
from llava.dataset.subtask.tools import convert_dct_list, give_order, total_valid, add_newline, intersection_area
from llava.dataset.subtask.crello import *

def convert_dct_list_crello_simple(layers):
    # bg, layers, *_ = pk
    psdsize = layers[0]['psd_size']
    # bgimg = download_image(bg)
    lys = []
    for x in layers:
        dct = {}
        dct['Angle'] = x['Angle']
        dct['Text'] = x['Text']
        dct['orgbox'] = x['orgbox']
        dct['category'] = x['label']
        dct['char_num'] = len(x['Text'].strip())
        dct['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], psdsize))
        dct['Bounding Box'] = x['Bounding Box']
        dct['fontsize'] = normalize_fontsize(x)
        dct['fontcolor'] = get_color(x['FillColor'])
        dct['img'] = x['img']

        lys.append(dct)
    return lys

def give_order_udl(pd):
    pd = sorted(pd, key=lambda x:x[:2][::-1])
    return pd
def convert(s, flip, bgimg):

    layers = convert_dct_list_crello_simple(s['layers'])
    layers = give_order(layers)

    lys = [x for x in layers if x['Text']!='' and not x.get('bad', False)]
    layers = lys
    ratio = np.random.rand()
    ratio = 0
    num = len(layers)
    done = min(int(num*ratio), num-1)
    
    udl = s['underlay']
    udl_bbox = []
    w, h = bgimg.width, bgimg.height
    for x in udl:
        x0,y0,x1,y1 = x
        if not flip:
            udl_bbox.append(json.loads(get_bbox_tokens(x, (w, h))))
        else:
            b2 = [w - x1+1, y0, w - x0-1, y1]
            udl_bbox.append(json.loads(get_bbox_tokens(b2, (w, h))))
    udl_bbox = give_order_udl(udl_bbox)


    # inpimg = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers[:done], flip).convert('RGB')

    # 其实也可以改成在这里多pop 
    [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('orgbox'), x.pop('Angle'),x.pop('fontcolor'),x.pop('fontsize'),x.pop('char_num')) for x in layers]
    # [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('orgbox'), x.pop('Angle')) for x in layers]
    iper_j = [{"category": x["category"]} for x in layers[done:]]
    # iper_j = add_newline(json.dumps([{"category": x["category"]} for x in layers[done:]]))

    dd = [x['bbox'] for x in layers[:done]]
    dd = give_order_udl(dd)
    dd = json.dumps(dd) 

    # 在这里放有点太靠后了 问题里没underlay了
    for bbox in udl_bbox:
        cur_item = {'category': 'underlay', 'bbox': bbox}
        layers.append(cur_item)

        iper_j.append({"category": "underlay"})

    iper = json.dumps(iper_j)

    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a poster image and a series of text to be added to the poster subsequently, predict the metadata for each text metadata listed below.\n''' + \
        f'''input: {iper}'''},
        {
            'from': 'gpt',
            'value': add_newline(json.dumps(layers[done:]))
        }
    ]
    return dialog, len(layers)