import json
from PIL import Image
import numpy as np
from .tools import convert_dct_list_text, give_order_text, total_valid, add_newline
from .unionfind import CustomUnionFind, do_overlap
from ..utils.download import download_image
from ..utils.imgproc import draw_all

def convert(pk, flip):
    bg = pk[0]
    bgimg = download_image(bg)
    layers = convert_dct_list_text(pk)
    layers = [x for x in layers if total_valid(x, bgimg.size)]
    uf = CustomUnionFind(hash_function=lambda x: (json.dumps(x), tuple(x['bbox'])), compare_function=lambda x, y: do_overlap(x[1],y[1]))
    uf.initialize(layers)
    grp = uf.groups()
    layers = [give_order_text([json.loads(y[0]) for y in x])[0] for x in grp]
    layers = give_order_text(layers)

    inpimg = draw_all(bgimg, layers, flip)
    [(x.pop('img'), x.pop("Bounding Box")) for x in layers]
    iper = add_newline(json.dumps([{"text": x["Text"] , "char_num":x["char_num"]} for x in layers]))
    res1 = add_newline(json.dumps([{"category": x["category"] , "char_num":x["char_num"]} for x in layers]))
    txt = [x['Text'] for x in layers]
    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given aa list of text, predict the corresponding category, bounding box, font size, font color, alignment for each element.The answer should be json format [{"text":xxx, "category": xxx, "char_num": xxx, "bbox": [x_min, y_min, x_max, y_max], "fontsize": xxx, "fontcolor": [red, green, blue], "alignment":xxx}, ...]'}. Think step by step''' + \
        f'''\ninput: {iper}'''},
        {
            'from': 'gpt',
            'value': f'Step 1: {res1}' + '\nStep 2: ' + add_newline(json.dumps(layers))
        }
    ]
    if flip:
        bgimg = Image.fromarray(np.array(bgimg)[:, ::-1])
    return dialog, bgimg# , txt