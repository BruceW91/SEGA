# import json
from PIL import Image
import numpy as np
from .unionfind import CustomUnionFind, do_overlap
from .tools import convert_dct_list, give_order, total_valid, add_newline, intersection_area
from .crello import *

from llava.dataset.zb_refine_noise_utils.add_noise_layout import total_add_noise
import copy
import simplejson as json

from llava.dataset.paint_bbox_utils.paint import visualize_output
import os


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

def get_cot_prompt(layers, underlays, temp_saliency_map):
    total_prompt = []
    origin_layers = layers.copy()

    #total_加噪声  最后给出三种类型吧 要不就弄三个列表 然后根据这三个列表生产prompt
    total_layouts, occlusion_errors, err_underlay, overlay_errors = total_add_noise(layers, underlays, temp_saliency_map)

    occlusion_errors_str = json.dumps([x['bbox'] for x in occlusion_errors]) 
    err_underlay_str = json.dumps(err_underlay)
    overlay_errors_str = str(overlay_errors)[1:-1]
    if len(overlay_errors)>0:
        total_prompt.append("Text elements are overlayed." + overlay_errors_str)
    if len(occlusion_errors)>0:
        total_prompt.append("Text elements are obscuring the important details in the background image." + occlusion_errors_str)
    if len(err_underlay)>0:
        total_prompt.append("Ensure that each underlay on the background image contains a text element."+ err_underlay_str)

    if len(total_prompt) == 0:
        total_prompt.append("The layout is fine.")
    return total_layouts ,total_prompt

def get_cot_prompt_no_detail(layers, underlays, temp_saliency_map):
    total_prompt = []
    origin_layers = layers.copy()

    #total_加噪声  最后给出三种类型吧 要不就弄三个列表 然后根据这三个列表生产prompt
    total_layouts, occlusion_errors, err_underlay, overlay_errors = total_add_noise(layers, underlays, temp_saliency_map)

    if len(origin_layers) != len(total_layouts):
        print('error!!!')

    occlusion_errors_str = json.dumps([x['bbox'] for x in occlusion_errors]) 
    err_underlay_str = json.dumps(err_underlay)
    overlay_errors_str = str(overlay_errors)[1:-1]
    if len(overlay_errors)>0:
        total_prompt.append("Text elements are overlayed.")
    if len(occlusion_errors)>0:
        total_prompt.append("Text elements are obscuring the important details in the background image.")
        total_prompt.append("Place the text elements in the blank space.")
    if len(err_underlay)>0:
        total_prompt.append("Ensure that each underlay on the background image contains a text element.")

    if len(total_prompt) == 0:
        total_prompt.append("The layout is fine.")
    return total_layouts ,total_prompt

# 对于加噪版本 temp_pred 是加噪声后自己给出来的  但我开头这里还是要穿进来 方便统一格式

# 基于这个没有噪声的版本写
def convert_no_score(s, flip, temp_saliency_map):
    
    # 我真吐了  这边还要改回相对坐标 查看是不是对齐了 ！！！
    # 分析之后 这个是正经的在没pad的图像上的绝对坐标
    psdsize = s['layers'][0]['psd_size']
    # json.loads(get_bbox_tokens_2(x['Bounding Box'], psdsize))

    layers = convert_dct_list_crello_simple(s['layers'])
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
    # udl_bbox =[{'category':'underlay', 'char_num':0,'bbox':x, 'fontsize':0} for x in udl_bbox]
    judl = json.dumps(udl_bbox)
    reduncy_layers = copy.deepcopy(layers)

    # inpimg = zb_draw_all_crello_for_readnoise((bgimg.shape[1], bgimg.shape[0]), bgimg, layers[:done], flip).convert('RGB')

    # 其实也可以改成在这里多pop 
    [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('orgbox'), x.pop('Angle'),x.pop('fontcolor'),x.pop('fontsize')) for x in layers]
    # [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('orgbox'), x.pop('Angle')) for x in layers]
    iper = add_newline(json.dumps([{"category": x["category"], "char_num":x["char_num"]} for x in layers[done:]]))
    # outer = add_newline(json.dumps([{"category": x["category"], "bbox":x["bbox"]} for x in layers[done:]]))
    dd = [x['bbox'] for x in layers[:done]]
    dd = give_order_udl(dd)
    dd = json.dumps(dd) 
    # 这边疑似把GT都给改了 ！！！
    
    cur_pred ,prompt_list = get_cot_prompt_no_detail(copy.deepcopy(layers), udl_bbox, temp_saliency_map)

    new_relative_list = [ [item['category'],item['bbox']]  for item in cur_pred ]
    new_absolute_list = [ [x[0],process_output(x[1], img_.size)] for x in new_relative_list ]

    # father_folder = 'infer_out_refine/debug_look2'
    # orig_f = os.path.join(father_folder, 'orig_img_noise')
    # cur_f  = os.path.join(father_folder, 'cur_img_noise')
    # os.makedirs(orig_f, exist_ok=True)
    # os.makedirs(cur_f, exist_ok=True)

    # in_orig_img = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, reduncy_layers, flip).convert('RGB')
    inpimg = visualize_output(new_absolute_list, bgimg)

    # in_orig_img.save(os.path.join(orig_f, s['layers'][0]['id'] + ".png"))
    # inpimg.save(os.path.join(cur_f, s['layers'][0]['id']+".png"))
    # inpimg = zb_draw_all_crello_for_readnoise(new_absolute_bboxes, (img_.size[1], img_.size[0]), bgimg, reduncy_layers, flip).convert('RGB')  # 全画上去


    [ x.pop('char_num') for x in cur_pred if 'char_num' in list(x.keys()) ]
    try:
        cur_pred_str = json.dumps(cur_pred)
    except Exception as e:
        print(e)
    # 下面对cur_pred 进行格式 转换 考虑相对坐标什么的

    total_str = ""
    for i in range(len(prompt_list)):
        total_str += f"{i}. {prompt_list[i]} "
    reasons = "Reasons:" + total_str
    # need_str = json.dumps(need_list)

    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a poster image ,a series of text to be added to the poster subsequently and a poster laypout, please refine the layouts and give reasons.\n''' + \
        f'''input: {iper}Layouts to be refined:" {cur_pred_str}'''},
        {
            'from': 'gpt',
            'value':f'underlay: {judl}\n'+ f'{reasons}\n'+ add_newline(json.dumps(layers[done:]))
        }
    ]
    return dialog, inpimg