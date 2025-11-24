# import json
from PIL import Image
import numpy as np
from .unionfind import CustomUnionFind, do_overlap
from .tools import convert_dct_list, give_order, total_valid, add_newline, intersection_area
from .crello import *

from llava.dataset.zb_refine_noise_utils.add_noise_layout import total_add_noise
import copy
import simplejson as json

from llava.dataset.zb_refine_noise_utils.zb_new_add_noise import add_noise_and_get_score_for_crello



def process_output(output,size):
    # 把相对左边变成绝对坐标   image.size
    max_size = max(list(size))
    min_size = min(list(size))
    size_list = list(size)
    diff = (max_size-min_size)//2
    min_id = size_list.index(min_size)

    output = [int(float(i)*max_size) for i in output]
    if min_id == 0: 
        output[0] = output[0]-diff
        output[2] = output[2]-diff
    else:
        output[1] = output[1]-diff
        output[3] = output[3]-diff

    return output

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

def get_cot_prompt_new(layers, underlays, temp_saliency_map, bgimg):
    total_prompt = []
    origin_layers = layers.copy()
    temp_saliency_map = Image.fromarray(temp_saliency_map)
    bgimg = Image.fromarray(bgimg)

    new_layers,scores = add_noise_and_get_score_for_crello(layers, underlays, temp_saliency_map, bgimg)
    scores_line = {'overlay':0.015,'utilization':0.076,'underlay_effectiveness_strict':0.95,'occlusion':0.55,'unreadability':0.037,'valid_score':0.98}
    # 针对cgl 来一个  夸张组 不止70%

    for key,score in scores_line.items():
        cur_line = score
        score_value = scores[key][0]
        if key == 'overlay' and cur_line < score_value:
            total_prompt.append("Text elements are overlayed.")
        if key == 'occlusion' and cur_line < score_value:
            total_prompt.append("Text elements are obscuring the important details in the background image.")
        if key == 'utilization' and cur_line > score_value:
            total_prompt.append("Place the text elements in the blank space.")
        if key == 'underlay_effectiveness_strict' and cur_line > score_value:
            total_prompt.append("Ensure that each underlay on the background image contains a text element.")
        # if key == 'valid_score' and cur_line > score_value:
        #     total_prompt.append("There should not be elements with overly small areas.")
        if key == 'unreadability' and cur_line < score_value:  
            total_prompt.append("Text elements should not be placed on backgrounds with drastic color changes.")

    if len(total_prompt) == 0:
        total_prompt.append("The layout is almost fine.")

    return new_layers ,total_prompt

# 对于加噪版本 temp_pred 是加噪声后自己给出来的  但我开头这里还是要穿进来 方便统一格式

# 基于这个没有噪声的版本写
def convert_no_score(s, flip, temp_saliency_map):
    
    psdsize = s['layers'][0]['psd_size']
    # json.loads(get_bbox_tokens_2(x['Bounding Box'], psdsize))

    layers = convert_dct_list_crello_simple(s['layers'])
    underlay_part = s['underlay']
    # if len(underlay_part) > 0:
    #     print("here")

    # 下面这个保证有序 很重要！！！
    layers = give_order(layers)
    img_ = get_bg_full(s['layers'][0]['psd_size'], s['bbox'], s['background'], layers)
    bgimg = np.array(img_)
    # inpimg = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers, flip).convert('RGB')
    lys = [x for x in layers if x['Text']!='' and not x.get('bad', False)]
    layers = lys

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


    reduncy_layers = copy.deepcopy(layers)

    # inpimg = zb_draw_all_crello_for_readnoise((bgimg.shape[1], bgimg.shape[0]), bgimg, layers[:done], flip).convert('RGB')

    # 其实也可以改成在这里多pop 
    [(x.pop('img'), x.pop('Text'), x.pop('orgbox'), x.pop('Angle'),x.pop('fontcolor'),x.pop('fontsize')) for x in layers]


    for item in layers:
        abs_bbox = item['Bounding Box']
        # r_bbox = json.loads(get_bbox_tokens_2(item['bbox'], bg_img.size))
        item['bbox'] = abs_bbox
    
    cur_pred ,prompt_list = get_cot_prompt_new(copy.deepcopy(layers), udl, temp_saliency_map, bgimg)
    bg_img = Image.fromarray(bgimg)
    new_absolute_bboxes = copy.deepcopy([x['bbox'] for x in cur_pred ]) 
    for item in cur_pred:
        r_bbox = json.loads(get_bbox_tokens_2(item['bbox'], bg_img.size))
        item['bbox'] = r_bbox

    inpimg = zb_draw_all_crello_for_readnoise(new_absolute_bboxes, (img_.size[1], img_.size[0]), bgimg, copy.deepcopy(reduncy_layers), flip).convert('RGB')  # 全画上去

    [(x.pop('img'), x.pop('Text'), x.pop('orgbox'), x.pop('Angle'),x.pop('fontcolor'),x.pop('fontsize'),x.pop('Bounding Box'),x.pop('char_num')) for x in reduncy_layers]
    [ (x.pop('char_num'), x.pop('Bounding Box')) for x in cur_pred if 'char_num' in list(x.keys()) ]
    # [ x.pop('bbox') for x in cur_pred]
    cur_pred =  sorted(cur_pred, key=lambda x: x['category'])
    try:
        cur_pred_str = json.dumps(cur_pred)
    except Exception as e:
        print(e)
    # 下面对cur_pred 进行格式 转换 考虑相对坐标什么的

    total_str = ""
    for i in range(len(prompt_list)):
        total_str += f"{i}. {prompt_list[i]} "
    reasons = "Evaluation:" + total_str

    for ubbox in udl_bbox:
        cur_item = {'category':'underlay','bbox':ubbox}
        reduncy_layers.append(cur_item)


    reduncy_layers = sorted(reduncy_layers, key=lambda x: x['category'])
    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a poster image and a poster laypout, please evaluate and refine the layouts and give reasons.\n''' + \
        f'''input: Layouts to be refined:" {cur_pred_str}'''},
        {
            'from': 'gpt',
            'value':f'{reasons}\n'+ add_newline(json.dumps(reduncy_layers[done:]))
        }
    ]
    return dialog, inpimg