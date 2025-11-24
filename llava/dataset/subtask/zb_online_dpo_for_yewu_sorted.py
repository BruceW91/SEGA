import json
from PIL import Image
import numpy as np
from .tools import convert_dct_list_text_buttons_for_yewu, give_order, get_bbox_tokens, total_valid, add_newline
from .unionfind import CustomUnionFind, do_overlap
from ..utils.download import download_image,download_image_for_yewu
from ..utils.imgproc import draw_all,draw_all_for_yewu


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
def perturb_bbox(bbox, sizes):
    bbox = translate_bbox(bbox, sizes)
    ratio = np.random.randn()
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

def simple_valid(x, sz):
    x0,y0,x1,y1 = bbox = x['Bounding Box']
    ww, hh = sz
    h, w = y1-y0, x1-x0
    if (w < ww and h < hh) and (x0 >=0 and x1<=ww and y0>=0 and y1<=hh) and (w*h>0):
        return True
    else:
        return False
def give_order_ocr(pd):
    pd = sorted(pd, key=lambda x:x['bbox'][:2][::-1])
    return pd

preferred_order = {'Title': 1, 'Subtitle': 2, 'Bodytext': 3}

def custom_sort_by_preferred_order(item):
    """
    根据预定义的顺序对字典列表进行排序。
    如果项不在预定义顺序中，则默认放在末尾。
    """
    name = item['category']
    # 如果名字在预定义顺序中，返回其对应的值作为排序键，否则返回一个很大的数保证其排在最后
    return preferred_order.get(name, float('inf'))

def custom_sort_by_preferred_order_s(item):
    """
    根据预定义的顺序对字典列表进行排序。
    如果项不在预定义顺序中，则默认放在末尾。
    """
    s_class = item.get('s_class', float('inf'))
    if type(s_class) == str:
        s_class = float(s_class.split('_')[0])

    # 如果名字在预定义顺序中，返回其对应的值作为排序键，否则返回一个很大的数保证其排在最后
    return s_class


def convert(pk, flip):
    bg = pk[0]
    bgimg = download_image_for_yewu(bg)
    ratio = np.random.rand()
    layers, _ = convert_dct_list_text_buttons_for_yewu(pk,bgimg.size)

    # for x in layers:
    #     s_class = x.get('s_class', float('inf'))
    #     if s_class == '2_1':
    #         print("jj")
    # 这里做排序
    layers = sorted(layers, key=custom_sort_by_preferred_order) # 默认稳定排序 第一遍弄title
    # print(layers)
    layers = sorted(layers, key=custom_sort_by_preferred_order_s)

    #debug


    gua = []
    but = []

    layers = [x for x in layers if simple_valid(x, bgimg.size)]
    # layers = give_order(layers)
    num = len(layers)
    origlayers = layers.copy()

    done = int(np.round(num*ratio))
    if done == num:
        done = num - 1

    #! 加入bbox位置扰动  
    orglayers = []
    for x in layers:
        y = {}

        for k, v in x.items():
            y[k] = v
        if True:  # 永久扰动
            # y['modified'] = True
            y['Bounding Box'], ratio = perturb_bbox(x['Bounding Box'], bgimg.size)
            y['bbox'] = json.loads(get_bbox_tokens(y['Bounding Box'], bgimg.size))

        orglayers.append(y)     # orglayers保存的视频原始信息

    done = 0
    bts = []
    
    inpimg = draw_all_for_yewu(bgimg, origlayers[:done], flip)

    [ x.pop("Bounding Box") for x in origlayers]
    [ x.pop("Bounding Box") for x in orglayers]

    temp_list = []
    for x in origlayers[done:]:
        cur = {}
        cur['category'] = x["category"]
        if 's_class' in x:
            cur['s_class'] = x["s_class"]
        if 'char_num' in x:
            cur['char_num'] = x["char_num"]
            # cur['line'] = x['line']
        temp_list.append(cur)
            
    iper = add_newline(json.dumps(temp_list)) 

    dder = [{'bbox':x['bbox']} for x in origlayers[:done]]
    dder = give_order_ocr(dder) #  这里有个按照ocr顺序
    dder = [x['bbox'] for x in dder]
    dd = add_newline(json.dumps(dder))
    dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a half-finished poster image and a series of text to be added to the poster subsequently, predict the metadata for each text metadata listed below.''' + \
        f'''input: {iper}'''},
        {
            'from': 'gpt',
            'value': f'previous: {dd}\nunderlay: {bts}\n' + add_newline(json.dumps(origlayers[done:]))
        }
    ]

    nega_dialog = [
        {'from': 'human',
   'value': '''<image>\n Given a half-finished poster image and a series of text to be added to the poster subsequently, predict the metadata for each text metadata listed below.''' + \
        f'''input: {iper}'''},
        {
            'from': 'gpt',
            'value': f'previous: {dd}\nunderlay: {bts}\n' + add_newline(json.dumps(orglayers[done:]))
        }
    ]    

    return dialog, nega_dialog, inpimg, done