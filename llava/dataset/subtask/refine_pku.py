import json
from PIL import Image
import numpy as np
from .tools import convert_dct_list_cgl, give_order_cgl, total_valid, add_newline, get_bbox_tokens
from .unionfind import CustomUnionFind, do_overlap
from ..utils.download import download_image
from ..utils.imgproc import draw_all_cgl
from io import BytesIO
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

def perturb_bbox(bbox, sizes):
    bbox = translate_bbox(bbox, sizes)
    ratio = np.random.randn()/1.5
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

def convert(pk, flip):
    if 'image' in pk.keys():
        bg = pk['image']
    else:
        bg = pk['inpainted_poster']
    bgimg = Image.open(BytesIO(bg))
    layers = convert_dct_list_cgl(pk)
    
    uf = CustomUnionFind(hash_function=lambda x: (json.dumps(x), tuple(x['bbox'])), compare_function=lambda x, y: False)
    uf.initialize(layers)
    grp = uf.groups()
    layers = [give_order_cgl([json.loads(y[0]) for y in x])[0] for x in grp]
    layers = give_order_cgl(layers)

    layers, inpimg = draw_all_cgl(bgimg, layers, flip)

    orglayers = []
    for x in layers:
        y = {}
        y['modified'] = True
        for k, v in x.items():
            y[k] = v
        if np.random.rand() > 0.3:
            y['modified'] = True
            
            x['Bounding Box'], ratio = perturb_bbox(x['Bounding Box'], bgimg.size)
            x["fontsize"] = np.around(x["fontsize"] * ratio, 3)
            x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], bgimg.size))
        else:
            y['modified'] = False
        orglayers.append(y)
    try:
        layers, inpimg = draw_all_cgl(bgimg, layers, flip)
        orglayers, _ = draw_all_cgl(bgimg, orglayers, flip)
    except Exception as e:
        print(e)
        raise e

    for x, y in zip(layers, orglayers):
        if not y['modified']:
            y['bbox'] = x['bbox']
    [(x.pop("Bounding Box"), x.pop("psd_size")) for x in layers]
    [(x.pop("Bounding Box"), x.pop("psd_size")) for x in orglayers]
    iper = add_newline(json.dumps(layers))
    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a poster image drawn with a series of text, and corresponding text metadata, you need to modify the problematic metadata within it to make the refined visuals reasonable and aesthetically pleasing. The answer should be written in json format. Each json should have a key “modified” indicating whether the original json need to be modified.''' + \
        f'''\ninput: {iper}'''},
        {
            'from': 'gpt',
            'value': add_newline(json.dumps(orglayers))
        }
    ]
    return dialog, inpimg
