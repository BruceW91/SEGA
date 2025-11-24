import json
from PIL import Image
import numpy as np
from .tools import convert_dct_list, give_order, total_valid, add_newline, get_bbox_tokens
from .unionfind import CustomUnionFind, do_overlap
from ..utils.download import download_image
from ..utils.imgproc import draw_all
import math

def check_position(bbox1, bbox2):
    # 计算两个bbox的中心点
    center1 = [(bbox1[0]+bbox1[2])/2, (bbox1[1]+bbox1[3])/2]
    center2 = [(bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2]

    # 计算中心点的相对位置
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    if dx*dy==0:
        return

    # 计算角度，转换为度数
    angle = math.atan2(-dy, dx) * 180 / math.pi
    # 判断中心点相对位置
    if -5 <= angle < 5:
        return 'rightwards'
    elif 5 <= angle < 85:
        return 'towards the top-right'
    elif 85 <= angle < 95:
        return 'topwards'
    elif 95 <= angle < 175:
        return 'towards the top-left'
    elif -175 <= angle < -95:
        return 'towards the bottom-left'
    elif -95 <= angle < -85:
        return 'downwards'
    elif -85 <= angle < -5:
        return 'towards the bottom-right'
    else:
        return 'leftwards'
    
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

char_size_dct = {
    'enormous':(1, 0.01),
    'very big':(0.01, 0.003),
    'big':(0.003, 0.0013),
    'normal':(0.0013, 0.0005),
    'small':(0.0005,0.00015),
    'very small': (0.00025,0.0),
}
char_cd = {
    'enormous':5,
    'very big':4,
    'big':3,
    'normal':2,
    'small':1,
    'very small': 0,
    }
def calc_area(bbox):
    x0, y0, x1, y1 = bbox
    return (x1-x0)*(y1-y0)

def calc_charsize(layer, sz):
    ratio = calc_area(layer['Bounding Box'])/np.multiply(*sz)/layer['char_num']
    for k, v in char_size_dct.items():
        if ratio<= v[0] and ratio > v[1]:
            return k
    a = calc_area(layer['Bounding Box'])
    b = sz
    c = layer['char_num']
    raise Exception(f'{a}\n{b}\n{c}\ninvisible charsize')

def relative_to_gt(inp, dct):
    org = dct[json.dumps(inp)]
    if org['modified']:
        return check_position(inp['bbox'], org['bbox'])
    
def rel_list(layers, dct):
    cot = ''
    last = ''
    for x in layers:
        res = relative_to_gt(x, dct)
        if x['category'] == last:
            end = ', '
            start = 'the next '
        else:
            if last == '':
                end = ''
            else:
                end = '; '
            start = 'The first '
        last = x['category']
        ft = x.get('ft', 'box size is good')
        if res is not None:
            cot+=(end +start + x['category'] + f' needs to be moved more {res} and {ft}')
        else:
            cot+=(end+ start + x['category'] + f' is good and {ft}')
    return cot+'. '

def overlap_detect(grp):
    sgrp = sorted(grp, key=lambda x:-len(x))
    cot = ''
    tps = []
    for x in sgrp:
        if len(x)<2:
            break
        else:
            ys = []
            for y in x:
                y = json.loads(y[0])
                newd = json.dumps({'category': y['category'],
                       'char_num': y['char_num'],
                       'bbox': y['bbox']})
                ys.append(newd)
            temp = ' and '.join(ys) + ' are overlapping. '
            tps.append(temp)
    if len(tps) == 0:
        return 'There is no overlapping. '
    elif len(tps) ==1:
        return f'Some texts are overlapped, they need to be separated: {tps[0]}'
    else:
        cot = f'Some texts are overlapped, they need to be separated: Firstly, {tps[0]}'
        for x in tps[1:]:
            cot += f'Then, {x}'
        return cot

def get_composition(layers, simple=False):
    # Initialize a 2D array to store the areas
    areas = [[0]*3 for _ in range(3)]

    # Define the function to update the area
    def update_area(bbox, areas):
        x1, y1, x2, y2 = bbox
        # Compute the width and height of the bbox
        width = x2 - x1
        height = y2 - y1
        # Determine which grid cells the bbox overlaps with
        for i in range(3):
            for j in range(3):
                # Compute the overlap with the grid cell
                overlap_x1 = max(x1, i / 3)
                overlap_y1 = max(y1, j / 3)
                overlap_x2 = min(x2, (i + 1) / 3)
                overlap_y2 = min(y2, (j + 1) / 3)
                # Update the area if there is an overlap
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    areas[i][j] += (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)

    # Use the function
    bboxes = [x['bbox'] for x in layers]  # Replace with your list of bboxes
    for bbox in bboxes:
        update_area(bbox, areas)
    dist = np.array(areas).transpose(1,0)
    dist = dist/dist.sum()
    hor = dist.sum(axis=0)
    indh = np.argsort(hor)
    ver = dist.sum(axis=1)
    indv = np.argsort(ver)
    # if simple:
    if hor[1] > 0.9:
        hors = 'center'
    elif abs(hor[0] - hor[2]) > 0.05:
        if hor[0] > hor[2]:
            hors = 'left'
        else:
            hors = 'right'
    else:
        hors = 'center'

    if ver[1] > 0.9:
        vers = 'center'
    elif abs(ver[0] - ver[2]) > 0.05:
        if ver[0] > ver[2]:
            vers = 'top'
        else:
            vers = 'bottom'
    else:
        vers = 'center'
    if simple:
        return f'{vers}-{hors}'
    
    if hor[indh[0]] > 0:
        hors = ['left', 'center', 'right'][indh[-1]]
    if ver[indv[0]] > 0:
        vers = ['top', 'center', 'bottom'][indv[-1]]
    if hor[indh[-1]]>0.5:
        if ver[indv[-1]]>0.5:
            fin= f'{vers}-{hors}'
        elif hor[indh[0]] > 0 and hor.shape[0]>=2:
            fin= f'vertical-{hors}'
        else:
            fin= f'{vers}-{hors}'
    else:
        if ver[indv[-1]]>0.5 and hor.shape[0]>=2 and ver[indv[0]] > 0:
            fin= f'{vers}-horizontal'
        elif ver[indv[-1]]>0.5:
            fin= f'{vers}-{hors}'
        else:
            if hor[indh[-1]] * ver[indv[-1]] <0.5:
                return 'balanced'
            else:
                return 'tight-center'
    if fin == 'center-center':
        if hor[indh[-1]] * ver[indv[-1]] <0.5:
            return 'balanced'
        else:
            return 'tight-center'
    else:
        return fin

def convert(pk, flip):
    bg = pk[0]
    bgimg = download_image(bg)
    layers = convert_dct_list(pk)
    layers = [x for x in layers if total_valid(x, bgimg.size)]
    uf = CustomUnionFind(hash_function=lambda x: (json.dumps(x), tuple(x['bbox'])), compare_function=lambda x, y: do_overlap(x[1],y[1]))
    uf.initialize(layers)
    grp = uf.groups()
    layers = [give_order([json.loads(y[0]) for y in x])[0] for x in grp]
    layers = give_order(layers)
    if len(layers) == 0:
        raise Exception('no valid layer')
    orglayers = []
    for x in layers:
        y = {}
        x['ft'] = 'box size is good'
        y['modified'] = True
        for k, v in x.items():
            y[k] = v
        if np.random.rand() > 0.3:
            y['modified'] = True
            orgft = calc_charsize(x, bgimg.size[:2])
            x['Bounding Box'], ratio = perturb_bbox(x['Bounding Box'], bgimg.size)
            x["fontsize"] = np.around(x["fontsize"] * ratio, 3)
            ft = calc_charsize(x, bgimg.size[:2])
            diff = char_cd[orgft] - char_cd[ft]
            if  diff > 1 :
                x['ft'] = f'the current boxsize is too small'
            elif diff == 1:
                x['ft'] = f'the current boxsize needs to be bigger'
            elif diff == 0 and ratio == 1:
                x['ft'] = f'the current boxsize is good'
            elif diff == -1:
                x['ft'] = f'the current boxsize needs to be smaller'
            elif diff < -1:
                x['ft'] = f'the current boxsize which is too big'
            elif ratio < 1:
                x['ft'] = f'the current boxsize needs to be a bit bigger'
            elif ratio > 1:
                x['ft'] = f'the current boxsize needs to be a bit smaller'
            x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], bgimg.size))
        else:
            y['modified'] = False
        orglayers.append(y)
    try:
        inpimg = draw_all(bgimg, layers, flip)
        _ = draw_all(bgimg, orglayers, flip)
    except Exception as e:
        print(e)
        raise e

    for x, y in zip(layers, orglayers):
        if not y['modified']:
            y['bbox'] = x['bbox']
    layout_cur = get_composition(layers)
    layout_org = get_composition(orglayers)
    
    dct = {json.dumps(x): y for x, y in zip(layers, orglayers)}
    uf = CustomUnionFind(hash_function=lambda x: (json.dumps(x), tuple(x['bbox'])), compare_function=lambda x, y: do_overlap(x[1],y[1]))
    uf.initialize(layers)
    grp = uf.groups()
    rel = rel_list(layers, dct)
    ovp = overlap_detect(grp)
    if layout_cur == layout_org:
        la = f'The current layout is {layout_cur} which is good. '
    else:
        la = f'The current layout {layout_cur} should be turned to {layout_org}. '

    cot = f'{la}{ovp}\n{rel}\n'
    
    [(x.pop('img'), x.pop("Bounding Box"), x.pop('ft')) for x in layers]
    [(x.pop('img'), x.pop("Bounding Box"), x.pop('ft')) for x in orglayers]
    iper = add_newline(json.dumps(layers))
    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a poster image drawn with a series of text, and corresponding text metadata, you need to modify the problematic(e.g. overlapping, wrong position, wrong bbox size and fontsize) metadata within it to make the refined visuals reasonable and aesthetically pleasing. The final answer should be written in json format. Each json should have a key “modified” indicating whether the original json need to be modified. Think step by step''' + \
        f'''\ninput: {iper}'''},
        {
            'from': 'gpt',
            'value': cot + add_newline(json.dumps(orglayers))
        }
    ]
    return dialog, inpimg