import json
from PIL import Image
import numpy as np
from .tools import convert_dct_list, give_order, total_valid, add_newline
from .crello import *
def give_order_udl(pd):
    pd = sorted(pd, key=lambda x:x[:2][::-1])
    return pd
def convert(s, flip):

    layers = convert_dct_list_crello(s['layers'])
    layers = give_order(layers)
    bgimg = np.array(get_bg_full(s['layers'][0]['psd_size'], s['bbox'], s['background'], layers))
    inpimg = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers, flip).convert('RGB')
    lys = [x for x in layers if x['Text']!='' and not x.get('bad', False)]
    layers = lys
    ratio = np.random.rand()
    num = len(layers)
    done = min(int(num*ratio), num-1)

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

    inpimg = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers[:done], flip).convert('RGB')
    _ = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers[done:], flip).convert('RGB')
    [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('orgbox'), x.pop('Angle')) for x in layers]
    iper = add_newline(json.dumps([{"category": x["category"], "char_num":x["char_num"]} for x in layers[done:]]))
    dd = [x['bbox'] for x in layers[:done]]
    dd = give_order_udl(dd)
    dd = json.dumps(dd) 
    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given only a background image, predict elements to be drawn in json format: [{"category": xxx, "bbox": [x_min, y_min, x_max, y_max]}, ...]''' + \
        f''''''},
        {
            'from': 'gpt',
            'value': f'previous: {dd}\nunderlay: {judl}\n'+add_newline(json.dumps(layers[done:]))
        }
    ]
    return dialog, inpimg