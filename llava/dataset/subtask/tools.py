import numpy as np
import json
from torch import Tensor

cate_dict = {x: i for i, x in enumerate(('title', 'subtitle', 'bodytext', 'calls to action', 'detailed items', 'menu items', 'social media', 'date', 'name', 'website', 'phone number', 'location'))}
cate_dict_cgl = {x: i for i, x in enumerate(('logo', 'text', 'underlay', 'embellishment'))}

def give_order(pd):
    pd = sorted(pd, key=lambda x:cate_dict.get(x['category'].lower(), 100))
    return pd

def give_order_text(pd):
    pd = sorted(pd, key=lambda x:-x['fontsize'])
    return pd
def convert_xywh_to_ltrb(bbox):
    assert len(bbox) == 4
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def _compute_iou(
    box_1,
    box_2,
    method = "iou",
):
    """
    Since there are many IoU-like metrics,
    we compute them at once and return the specified one.
    box_1 and box_2 are in (N, 4) format.
    """
    assert method in ["iou", "giou", "ai/a1", "ai/a2"]

    if isinstance(box_1, Tensor):
        box_1 = np.array(box_1)
        box_2 = np.array(box_2)
    assert len(box_1) == len(box_2)

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = np.maximum(l1, l2)
    r_min = np.minimum(r1, r2)
    t_max = np.maximum(t1, t2)
    b_min = np.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    if method == "iou":
        return iou
    elif method == "ai/a1":
        return ai / a1
    elif method == "ai/a2":
        return ai / a2

    # outer region
    l_min = np.minimum(l1, l2)
    r_max = np.maximum(r1, r2)
    t_min = np.minimum(t1, t2)
    b_max = np.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou: np.ndarray = iou - (ac - au) / ac

    return giou

def classify_layers(layers):
    classified_layers = {}
    classified_layers['logo'] = []
    classified_layers['text'] = []
    classified_layers['embellishment'] = []
    classified_layers['underlay'] = []
    
    for ly in layers:
        if ly['category'] == 'logo':
            classified_layers['logo'].append(ly)
        if ly['category'] == 'text':
            classified_layers['text'].append(ly)
        if ly['category'] == 'underlay':
            classified_layers['underlay'].append(ly)
        if ly['category'] == 'embellishment':
            classified_layers['embellishment'].append(ly)
    return classified_layers

def get_cen_w_h(pred_bbox):
    w, h = (pred_bbox[2]-pred_bbox[0]), (pred_bbox[3] - pred_bbox[1])
    cx, cy = (pred_bbox[0] + w//2), (pred_bbox[1] + h//2)
    return [cx,cy,w,h]

def give_order_cgl(pd, underlay_order=False):
    pd = sorted(pd, key=lambda x:cate_dict_cgl.get(x['category'].lower(), 100))
    if underlay_order:
        classified_layers = classify_layers(pd)
        #to_delect_layer = []
        for k, v in classified_layers.items():
            classified_layers[k] = sorted(v, key=lambda x:x['Bounding Box'][:2][::-1])
            if k == 'underlay':
                groups = {}
                logo_text_emb = classified_layers['logo']+classified_layers['text'] + classified_layers['embellishment']
                for uid, u in enumerate(classified_layers[k]):
                    groups[uid] = []
                    #print(uid)
                    for inst in logo_text_emb:
                        
                        box2 = get_cen_w_h(inst['Bounding Box'])
                        box1 = get_cen_w_h(u['Bounding Box'])
                        iou = _compute_iou(np.array(box1)[np.newaxis,:],np.array(box2)[np.newaxis,:],method='ai/a2')[0]#cal_iou(box1,box2).tolist()[0][0]
                        #print(iou)
                        thresh = 1.0 - np.finfo(np.float32).eps
                        if iou >= thresh:
                            groups[uid].append(inst)

        ranked_layers = classified_layers['underlay']
        for inst in logo_text_emb:
            count = 0
            for k,v in groups.items():
                if inst in v:
                    inst['corresponding underlay'] = ranked_layers[k]['bbox']
                else:
                    count += 1
            if count == len(list(groups.keys())):
                inst['corresponding underlay'] = 'None'
            bbox = inst.pop('bbox')
            inst['bbox'] = bbox
        for inst in ranked_layers:
            inst['corresponding underlay'] = 'None'
            bbox = inst.pop('bbox')
            inst['bbox'] = bbox
            

        # for _i, inst in enumerate(classified_layers['logo']):
        #     inst['category'] = 'logo'+str(_i+1)
        # for _i, inst in enumerate(classified_layers['text']):
        #     inst['category'] = inst['category']+str(_i+1)
        # for _i, inst in enumerate(classified_layers['embellishment']):
        #     inst['category'] = inst['category']+str(_i+1)

        # logo_text_emb = classified_layers['logo']+classified_layers['text'] + classified_layers['embellishment']

        # ranked_layers = classified_layers['underlay']
        # for u in ranked_layers:
        #     u['included_elements'] = []
        # for inst in logo_text_emb:
        #     count = 0
        #     for k,v in groups.items():
        #         if inst in v:
        #             ranked_layers[k]['included_elements'].append(inst['category'])
        
        # ranked_layers = logo_text_emb + ranked_layers
                    
        ranked_layers = ranked_layers+logo_text_emb
        return ranked_layers
    else:
        return pd

def add_newline(jres):
    text = ''
    for x in jres:
        if x == '}':
            text += '}\n'
        else:
            text += x
    return text

def total_valid(x, sz):
    x0,y0,x1,y1 = bbox = x['Bounding Box']
    ww, hh = sz
    h, w = y1-y0, x1-x0
    if (w < ww and h < hh) and (x0 >=0 and x1<=ww and y0>=0 and y1<=hh) and (w >= h) and (w*h>0):
        return True
    else:
        return False
    
def total_valid_qtx(x, sz):
    x0,y0,x1,y1 = bbox = x['Bounding Box']
    ww, hh = sz
    h, w = y1-y0, x1-x0
    if (w < ww and h < hh) and (x0 >=0 and x1<=ww and y0>=0 and y1<=hh):
        return True
    else:
        return False
    
def get_color(color):
    return str([np.around(float(x), 2) for x in color[:3]])

def normalize_fontsize(layer):
    fontsize = layer['FontSize']
    sizes = layer['psd_size']

    width, height = sizes[0],sizes[1]
    if width == height:
        fontsize = np.around(fontsize / sizes[1],3)
    elif width > height:
        fontsize = np.around(fontsize / sizes[0],3)
    else:
        fontsize = np.around(fontsize / sizes[1],3)

    return fontsize

def get_bbox_tokens(bbox,sizes):
    ratio = np.array(sizes).max()
    bbox = list(bbox)
    #print(ratio,sizes)
    width, height = sizes[0],sizes[1]
    if width == height:
        return str(np.around(np.array(bbox)/ratio,3).tolist())
    elif width > height:
        offset = (width - height) // 2
        bbox = list(bbox)
        bbox[1] += offset
        bbox[3] += offset
        return str(np.around(np.array(bbox)/ratio,3).tolist())
    else:
        offset = (height - width) // 2
        bbox[0] += offset
        bbox[2] += offset
        return str(np.around(np.array(bbox)/ratio,3).tolist())
    
def convert_dct_list(pk):
    bg, layers, *_ = pk
    psdsize = layers[0]['psd_size']
    # bgimg = download_image(bg)
    lys = []
    for x in layers:
        dct = {}
        dct['category'] = x['label']
        dct['char_num'] = len(x['Text'].strip())
        dct['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], psdsize))
        dct['Bounding Box'] = x['Bounding Box']
        dct['fontsize'] = normalize_fontsize(x)
        dct['fontcolor'] = get_color(x['FillColor'])
        dct['img'] = x['img']
        num = x['Justification']
        if num == 0 or num == 3 or num==5:
            dct['alignment'] = 'left'
        elif num == 1:
            dct['alignment'] = 'right'
        else:
            dct['alignment'] = 'center'
        lys.append(dct)
    return lys

def convert_dct_list_cot(pk):
    bg, layers, _ = pk
    psdsize = layers[0]['psd_size']
    # bgimg = download_image(bg)
    lys = []
    for x in layers:
        dct = {}
        dct['thought'] = ''
        dct['category'] = x['label']
        dct['char_num'] = len(x['Text'].strip())
        dct['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], psdsize))
        dct['Bounding Box'] = x['Bounding Box']
        dct['fontsize'] = normalize_fontsize(x)
        dct['fontcolor'] = get_color(x['FillColor'])
        dct['img'] = x['img']
        num = x['Justification']
        if num == 0 or num == 3 or num==5:
            dct['alignment'] = 'left'
        elif num == 1:
            dct['alignment'] = 'right'
        else:
            dct['alignment'] = 'center'
        lys.append(dct)
    return lys

def calc_area(bbox):
    x0, y0, x1, y1 = bbox
    return (x1-x0+1) * (y1-y0+1)

def intersection_area(bbox1, bbox2):
    # 计算相交区域的坐标
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    # 计算相交区域的宽度和高度
    width = max(0, x1 - x0 + 1)
    height = max(0, y1 - y0 + 1)

    # 计算相交区域的面积
    area = width * height
    return area

def text_in(textbbox, underbbox, thres):
    inter_area = intersection_area(textbbox, underbbox)
    ratio = inter_area / calc_area(textbbox)
    return ratio > thres

def exists_text_in(underbbox, tlayers, thres=0.95):
    fin = []
    for i, x in enumerate(tlayers):
        bbox = x['Bounding Box']
        if text_in(bbox, underbbox, thres):
            fin.append(i)
    return fin

def convert_dct_list_text_buttons(pk):
    # bg, layers, _ = pk
    # buttons = []
    bg, layers, _, buttons = pk
    bt = { i: [] for i in range(len(layers))}
    for x in buttons:
        res = exists_text_in(x, layers)
        for y in res:
            x = [int(z) for z in x]
            bt[y].append(x)

    psdsize = layers[0]['psd_size']
    w, h = psdsize
    gua = []
    for but in buttons:
        # x0, y0, x1, y1 = [int(_) for _ in but]
        gua.append(json.loads(get_bbox_tokens([int(_) for _ in but], (w, h))))
    # bgimg = download_image(bg)
    lys = []
    for i, x in enumerate(layers):
        dct = {}
        dct['buttons'] = bt[i]
        dct['thought'] = ''
        if x['label'].lower() != 'bodytext':
            dct['Text'] = x['Text']  
        else:
            a = x['Text'][:100]
            b = x['Text'][100:]
            dct['Text'] = a + b
        dct['category'] = x['label']
        dct['char_num'] = len(dct['Text'].strip())
        dct['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], psdsize))
        dct['Bounding Box'] = x['Bounding Box']
        dct['fontsize'] = normalize_fontsize(x)
        dct['fontcolor'] = get_color(x['FillColor'])
        dct['img'] = x['img']
        num = x['Justification']
        if num == 0 or num == 3 or num==5:
            dct['alignment'] = 'left'
        elif num == 1:
            dct['alignment'] = 'right'
        else:
            dct['alignment'] = 'center'
        # dct['buttons'] = [json.loads(get_bbox_tokens(z, psdsize)) for z in dct['buttons']]
        lys.append(dct)
    return lys, gua

def normalize_fontsize_for_yewu(layer,psdsize):
    fontsize = layer['FontSize']
    sizes = psdsize

    width, height = sizes[0],sizes[1]
    if width == height:
        fontsize = np.around(fontsize / sizes[1],3)
    elif width > height:
        fontsize = np.around(fontsize / sizes[0],3)
    else:
        fontsize = np.around(fontsize / sizes[1],3)

    return fontsize

def convert_dct_list_text_buttons_for_yewu(pk,psdsize):
    bg, layers, _ = pk
    buttons = []

    w, h = psdsize
    gua = []
    # for but in buttons:
    #     # x0, y0, x1, y1 = [int(_) for _ in but]
    #     gua.append(json.loads(get_bbox_tokens([int(_) for _ in but], (w, h))))
    # bgimg = download_image(bg)
    lys = []
    for i, x in enumerate(layers):
        dct = {}
        if 's_class' in x:
            dct['s_class'] = x['s_class']
        dct['category'] = x['label']
        dct['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], psdsize))
        dct['Bounding Box'] =  [ int(t) for t in x['Bounding Box'] ]     # 改了
        
        if x['label'] == 'Media':
            pass  # 这里之前写错了
        else:
            dct['fontsize'] = normalize_fontsize_for_yewu(x,psdsize)
            if 'char_num' in x:
                dct['char_num'] = x['char_num']
            dct['fontcolor'] = x['FillColor']
            if 'line' in x:
                dct['line'] = x['line']
            # dct['char_num']

        lys.append(dct)
    return lys, gua

def convert_dct_list_text_buttons_for_yewu_human(pk,psdsize):
    bg, layers, _ = pk
    buttons = []

    w, h = psdsize
    gua = []
    # for but in buttons:
    #     # x0, y0, x1, y1 = [int(_) for _ in but]
    #     gua.append(json.loads(get_bbox_tokens([int(_) for _ in but], (w, h))))
    # bgimg = download_image(bg)
    lys = []
    for i, x in enumerate(layers):
        dct = {}
        dct['category'] = x['label']
        dct['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], psdsize))
        dct['Bounding Box'] =  [ int(t) for t in x['Bounding Box'] ]     # 改了

        lys.append(dct)
    return lys, gua

def convert_dct_list_text_buttons_for_0530(pk,psdsize):
    bg, layers, _ = pk
    buttons = []

    w, h = psdsize
    gua = []
    # for but in buttons:
    #     # x0, y0, x1, y1 = [int(_) for _ in but]
    #     gua.append(json.loads(get_bbox_tokens([int(_) for _ in but], (w, h))))
    # bgimg = download_image(bg)
    lys = []
    for i, x in enumerate(layers):
        dct = {}
        dct['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], psdsize))
        dct['Bounding Box'] =  [ int(t) for t in x['Bounding Box'] ]     # 改了
        dct['category'] = x['label']
        if x['label'] == 'Media':
            continue
        else:
            dct['fontsize'] = normalize_fontsize_for_yewu(x,psdsize)
            dct['char_num'] = x['char_num']
            dct['len'] = x['lens']
            # dct['char_num']
        lys.append(dct)
    return lys, gua

def convert_dct_list_text(pk):
    bg, layers, *_ = pk
    psdsize = layers[0]['psd_size']
    # bgimg = download_image(bg)
    lys = []
    for x in layers:
        dct = {}
        # dct['thought'] = ''
        if x['label'].lower() != 'bodytext':
            dct['Text'] = x['Text']  
        else:
            a = x['Text'][:100]
            b = x['Text'][100:]
            dct['Text'] = a + b
        dct['category'] = x['label']
        dct['char_num'] = len(dct['Text'].strip())
        dct['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], psdsize))
        dct['Bounding Box'] = x['Bounding Box']
        dct['fontsize'] = normalize_fontsize(x)
        dct['fontcolor'] = get_color(x['FillColor'])
        dct['img'] = x['img']
        num = x['Justification']
        if num == 0 or num == 3 or num==5:
            dct['alignment'] = 'left'
        elif num == 1:
            dct['alignment'] = 'right'
        else:
            dct['alignment'] = 'center'
        lys.append(dct)
    return lys

def convert_dct_list_qtx(pk):
    bg, layers, _ = pk
    psdsize = layers[0]['psd_size']
    # bgimg = download_image(bg)
    lys = []
    for x in layers:
        dct = {}
        dct['psd_size'] = psdsize
        dct['category'] = x['label']
        dct['char_num'] = len(x['Text'].strip())
        dct['text'] = x['Text']
        dct['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], psdsize))
        dct['Bounding Box'] = x['Bounding Box']
        dct['fontsize'] = normalize_fontsize(x)
        dct['fontcolor'] = get_color(x['FillColor'])
        dct['img'] = x['img']
        num = x['Justification']
        if num == 0 or num == 3 or num==5:
            dct['alignment'] = 'left'
        elif num == 1:
            dct['alignment'] = 'right'
        else:
            dct['alignment'] = 'center'
        lys.append(dct)
    return lys

def convert_dct_list_cgl(pk):
    layers = pk['layers']
    psdsize = layers[0]['psd_size']
    # bgimg = download_image(bg)
    lys = []
    for x in layers:
        dct = {}
        dct['category'] = x['label']
        if 'underlay' in x.keys():
            dct['underlay'] = x['underlay']
        dct['psd_size'] = x['psd_size']
        dct['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], psdsize))
        dct['Bounding Box'] = x['Bounding Box']
        lys.append(dct)
    return lys