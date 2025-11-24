import json
from PIL import Image
import numpy as np
from .tools import convert_dct_list, give_order, total_valid, add_newline
from .crello import *

def convert(s, flip):

    layers = convert_dct_list_crello(s['layers'])
    layers = give_order(layers)
    # lys = [x for x in layers if x['Text']!='']
    bgimg = np.array(get_bg_full(s['layers'][0]['psd_size'], s['bbox'], s['background'], layers))
    inpimg = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers, flip).convert('RGB')
    lys = [x for x in layers if x['Text']!='' and not x.get('bad', False)]
    layers = lys
    [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('orgbox'), x.pop('Angle')) for x in layers]
 
    [(x.pop("char_num"), x.pop("fontsize"), x.pop("fontcolor"), x.pop("alignment")) for x in layers]
    iper = add_newline(json.dumps([{"category": x["category"]} for x in layers]))
    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a background image and a list of elements only with text category, predict the corresponding bounding box only. The answer should be json format [{"category": xxx, "bbox": [x_min, y_min, x_max, y_max]}, ...]''' + \
        f'''\ninput: {iper}'''},
        {
            'from': 'gpt',
            'value': add_newline(json.dumps(layers))
        }
    ]
    if flip:
        bgimg = Image.fromarray(np.array(bgimg)[:, ::-1])
    else:
        bgimg = Image.fromarray(np.array(bgimg))
    return dialog, bgimg