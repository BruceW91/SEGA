import json
from PIL import Image
import numpy as np
from .unionfind import CustomUnionFind, do_overlap
from .tools import convert_dct_list, give_order, total_valid, add_newline, intersection_area
from .crello import *

def convert_dct_list_crello_simple(layers):
    # bg, layers, *_ = pk
    psdsize = layers[0]['psd_size']
    # bgimg = download_imc reage(bg)
    lys = []
    for x in layers:
        dct = {}
        dct['Angle'] = x['Angle']
        dct['Text'] = x['Text']
        dct['orgbox'] = x['orgbox']
        dct['category'] = x['label']
        dct['char_num'] = len(x['Text'].strip())
        dct['bbox'] = json.loads(get_bbox_tokens_2(x['Bounding Box'], psdsize))
        dct['Bounding Box'] = x['Bounding Box']
        dct['fontsize'] = normalize_fontsize(x)
        dct['fontcolor'] = get_color(x['FillColor'])
        dct['img'] = x['img']

        lys.append(dct)
    return lys

def give_order_udl(pd):
    pd = sorted(pd, key=lambda x:x[:2][::-1])
    return pd
def convert(s, flip, ratio_0 = False):

    layers = convert_dct_list_crello_simple(s['layers'])
    layers = give_order(layers)
    bgimg = np.array(get_bg_full(s['layers'][0]['psd_size'], s['bbox'], s['background'], layers))
    # inpimg = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers, flip).convert('RGB')
    lys = [x for x in layers if x['Text']!='' and not x.get('bad', False)]
    layers = lys
    ratio = np.random.rand()
    if ratio_0:
        ratio = 0
    num = len(layers)
    done = min(int(num*ratio), num-1)
    
    udl = s['underlay']
    udl_bbox = []
    w, h = bgimg.shape[1], bgimg.shape[0]
    for x in udl:
        x0,y0,x1,y1 = x
        if not flip:
            udl_bbox.append(json.loads(get_bbox_tokens_2(x, (w, h))))
        else:
            b2 = [w - x1+1, y0, w - x0-1, y1]
            udl_bbox.append(json.loads(get_bbox_tokens_2(b2, (w, h))))
    udl_bbox = give_order_udl(udl_bbox)
    # udl_bbox =[{'category':'underlay', 'char_num':0,'bbox':x, 'fontsize':0} for x in udl_bbox]
    judl = json.dumps(udl_bbox)

    inpimg = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers[:done], flip).convert('RGB')

    # 其实也可以改成在这里多pop 
    [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('orgbox'), x.pop('Angle'),x.pop('fontcolor'),x.pop('fontsize')) for x in layers]
    # [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('orgbox'), x.pop('Angle')) for x in layers]
    iper = add_newline(json.dumps([{"category": x["category"], "char_num":x["char_num"]} for x in layers[done:]]))
    # outer = add_newline(json.dumps([{"category": x["category"], "bbox":x["bbox"]} for x in layers[done:]]))
    dd = [x['bbox'] for x in layers[:done]]
    dd = give_order_udl(dd)
    dd = json.dumps(dd) 

    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a half-finished poster image , underlay location  and a series of text to be added to the poster subsequently, predict the metadata for each text metadata listed below and we need to make sure there are texts inside the underlay.\n''' + \
        f'''underlay: {judl},input: {iper}'''},
        {
            'from': 'gpt',
            'value': f'previous: {dd}\n'+add_newline(json.dumps(layers[done:]))
        }
    ]
    return dialog, inpimg, len(layers)