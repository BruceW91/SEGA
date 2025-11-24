import json
from PIL import Image
import numpy as np
from .unionfind import CustomUnionFind, do_overlap
from .tools import convert_dct_list, give_order, total_valid, add_newline, intersection_area
from .crello import *

def give_order_udl(pd):
    pd = sorted(pd, key=lambda x:x[:2][::-1])
    return pd
def convert(s, flip, ratio_0=False):
    # 这里把最后拼对话字符串单作为一个函数 就结偶了

    layers = convert_dct_list_crello_get_bbox_tokens_2(s['layers'])
    layers = give_order(layers)
    bgimg = np.array(get_bg_full(s['layers'][0]['psd_size'], s['bbox'], s['background'], layers))
    inpimg = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers, flip).convert('RGB')
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
            udl_bbox.append(json.loads(get_bbox_tokens(x, (w, h))))
        else:
            b2 = [w - x1+1, y0, w - x0-1, y1]
            udl_bbox.append(json.loads(get_bbox_tokens(b2, (w, h))))
    udl_bbox = give_order_udl(udl_bbox)
    # udl_bbox =[{'category':'underlay', 'char_num':0,'bbox':x, 'fontsize':0} for x in udl_bbox]
    judl = json.dumps(udl_bbox)


    inpimg = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers[:done], flip).convert('RGB')
    # _ = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers[done:], flip).convert('RGB')
    [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('orgbox'), x.pop('Angle'), x.pop('fontsize'),x.pop('fontcolor'),x.pop('alignment')) for x in layers]
    iper = add_newline(json.dumps([{"category": x["category"], "char_num":x["char_num"]} for x in layers[done:]]))
    # outer = add_newline(json.dumps([{"category": x["category"], "bbox":x["bbox"]} for x in layers[done:]]))
    dd = [x['bbox'] for x in layers[:done]]
    dd = give_order_udl(dd)
    dd = json.dumps(dd) 

    return [done,iper,dd,judl,layers], inpimg

def convert_and_merge_2_dialog(s, flip, example_s,cot_flag = False):

    items, img = convert(s, flip, cot_flag)
    e_items, e_img = convert(example_s, flip, True)

    done,iper,dd,judl,layers = items 
    e_done,example_iper,e_dd,e_judl,e_layers = e_items 
    tt = f''' Example, input: <image>\n {example_iper}''' + f'''value: '''

    dialog = [
        {'from': 'human',
   'value': '''Given a half-finished poster image and a series of text to be added to the poster subsequently, predict the metadata for each text metadata listed below. Firstly, learn an example and then provide an answer\n''' + \
        tt + add_newline(json.dumps(e_layers[e_done:]))  +  f''' Question, input: <image>\n {iper}'''},
        {
            'from': 'gpt',
            'value': f'previous: {dd}\nunderlay: {judl}\n'+add_newline(json.dumps(layers[done:]))
        }
    ]    
    return dialog, img, e_img