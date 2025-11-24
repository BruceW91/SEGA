import math
import numpy as np
import cv2
from PIL import Image
import traceback

from llava.dataset.bigposter.offline_utils import get_my_poster_png

class WJHException(Exception):
    def __init__(self, message):
        self.message = message
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
    
def get_bbox_tokens_2(bbox,sizes):
    round_l = 2
    ratio = np.array(sizes).max()
    bbox = list(bbox)
    #print(ratio,sizes)
    width, height = sizes[0],sizes[1]
    if width == height:
        return str(np.around(np.array(bbox)/ratio,round_l).tolist())
    elif width > height:
        offset = (width - height) // 2
        bbox = list(bbox)
        bbox[1] += offset
        bbox[3] += offset
        return str(np.around(np.array(bbox)/ratio,round_l).tolist())
    else:
        offset = (height - width) // 2
        bbox[0] += offset
        bbox[2] += offset
        return str(np.around(np.array(bbox)/ratio,round_l).tolist())

def get_bbox_tokens_2_wo_reshape(bbox,sizes):
    round_l = 2
    ratio = np.array(sizes).max()
    bbox = list(bbox)
    #print(ratio,sizes)
    width, height = sizes[0],sizes[1]
    return str(np.around(np.array(bbox)/ratio,round_l).tolist())

def resize_pad(img, k):
    dw, dh = k
    oh, ow = img.shape[:2]
    rh = dh/oh
    rw = dw/ow
    ratio = min(rw, rh)
    th, tw = int(np.floor(ratio *oh)), int(np.floor(ratio*ow))
    imgtemp = cv2.resize(img, (tw,th), fx=ratio, fy=ratio)
    up = (dh - th)//2
    down = dh-th-up
    left = (dw - tw)//2
    right = dw-tw-left
    temp = np.zeros((dh, dw, 4))
    if img.shape[-1] == 4:
        temp[up:up+th, left:left+tw] = imgtemp
    else:
        temp[up:up+th, left:left+tw, :3] = imgtemp
        temp[up:up+th, left:left+tw, 3] = 255
    return temp.astype(np.uint8)


def get_bbox_from_gray(gray_image):
    image_array = np.array(gray_image)

    # Find the coordinates where the pixel values are greater than 128
    coordinates = np.argwhere(image_array > 0)

    # If there are no such coordinates, the bounding box is undefined
    if coordinates.shape[0] == 0:
        bbox = None
    else:
        # Calculate the bounding box (min_row, min_col, max_row, max_col)
        min_row, min_col = np.min(coordinates, axis=0)
        max_row, max_col = np.max(coordinates, axis=0)
        bbox = np.array([min_col, min_row, max_col+1, max_row+1])
    return bbox
def rotate_bbox(bbox, theta):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2  # 计算矩形框的中心点

    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # 原始的四个顶点
    corners_rot = []  # 存储旋转后的四个顶点

    for (x, y) in corners:
        # 计算旋转后的坐标
        x_rot = cx + math.cos(theta) * (x - cx) - math.sin(theta) * (y - cy)
        y_rot = cy + math.sin(theta) * (x - cx) + math.cos(theta) * (y - cy)
        corners_rot.append((x_rot, y_rot))

    return corners_rot
def warpPers(fg, orgbox, sz, angle):
    h, w = fg.shape[:2]
    boxer = bbox_real(orgbox, sz)
    # 构造前景图像的四个顶点（左上，右上，右下，左下）
    fg_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    corners = rotate_bbox(bbox_real(orgbox, sz), angle)
    # 构造变换矩阵
    M = cv2.getPerspectiveTransform(fg_pts, np.float32(corners))
    warped_fg = cv2.warpPerspective(fg, M, sz)
    return warped_fg, corners

def put_text_bg_crello(imginp, bbox, bg):
    bg = np.array(bg)
    mx0, my0, mx1, my1 = bbox
    # try:
    #     imgnper = resize_pad(np.array(imginp), (mx1 - mx0+1, my1 - my0+1))
    # except:
    #     raise WJHException(f"图片尺寸过大 {str(bbox)}")
    # imgnper = resize_pad(np.array(imginp), (mx1 - mx0+1, my1 - my0+1))
    imgnper = imginp

    etax0 = -mx0 if mx0 < 0 else 0
    etay0 = -my0 if my0 < 0 else 0
    # dsth, dstw = imgnp[my0:my1, mx0:mx1].shape[:2]     
    dsth, dstw = bg[max(0, my0):my1, max(0, mx0):mx1].shape[:2] 
    # print(my1-etay0, mx1-etax0)
    ka = imgnper[etay0:etay0+dsth, etax0:etax0+dstw]
    mka = ka[..., 3:]/255
    msk = bg[max(0, my0):my1, max(0, mx0):mx1][...,3:]
    kaba = ka[..., :3] * mka + bg[max(0, my0):my1, max(0, mx0):mx1, :3] * (1-mka)
    org = bg[max(0, my0):my1, max(0, mx0):mx1, :3].copy()
    bg[max(0, my0):my1, max(0, mx0):mx1, :3] = kaba
    rev = False
    # return org, bg[max(0, my0):my1, max(0, mx0):mx1, :3], mka
    ow = org[mka[...,0]>0.1]
    dw = bg[max(0, my0):my1, max(0, mx0):mx1, :3][mka[...,0]>0.1]
    # print(max(abs((ow*1. - dw)).mean(axis=0)))
    if max(abs((ow*1. - dw*1.)).mean(axis=0)) < 10:
        rev = True
        ka[..., :3] = 255 - ka[..., :3]
        kaba = ka[..., :3] * mka + bg[max(0, my0):my1, max(0, mx0):mx1, :3] * (1-mka)
        org = bg[max(0, my0):my1, max(0, mx0):mx1, :3].copy()
        bg[max(0, my0):my1, max(0, mx0):mx1, :3] = kaba
    return bg, rev

from llava.dataset.subtask.tools import *
from PIL import Image
from io import BytesIO
def convert_dct_list_crello(layers):
    # bg, layers, *_ = pk
    psdsize = layers[0]['psd_size']
    # bgimg = download_image(bg)
    lys = []
    for x in layers:
        dct = {}
        dct['Angle'] = x['Angle']
        dct['Text'] = x['Text']
        dct['orgbox'] = x['orgbox']
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

def convert_dct_list_crello_get_bbox_tokens_2(layers):
    # bg, layers, *_ = pk
    psdsize = layers[0]['psd_size']
    # bgimg = download_image(bg)
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
        num = x['Justification']
        if num == 0 or num == 3 or num==5:
            dct['alignment'] = 'left'
        elif num == 1:
            dct['alignment'] = 'right'
        else:
            dct['alignment'] = 'center'
        lys.append(dct)
    return lys

def convert_dct_list_crello_get_bbox_tokens_2_wo_reshapebbox(layers):
    # bg, layers, *_ = pk
    psdsize = layers[0]['psd_size']
    # bgimg = download_image(bg)
    lys = []
    for x in layers:
        dct = {}
        dct['Angle'] = x['Angle']
        dct['Text'] = x['Text']
        dct['orgbox'] = x['orgbox']
        dct['category'] = x['label']
        dct['char_num'] = len(x['Text'].strip())
        dct['bbox'] = json.loads(get_bbox_tokens_2_wo_reshape(x['Bounding Box'], psdsize))
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

def convert_dct_list_new_crello(layers):
    # bg, layers, *_ = pk
    psdsize = layers[0]['psd_size']
    # bgimg = download_image(bg)
    lys = []
    for x in layers:
        temp_bbox = x.get('new_Bounding_Box', x.get('Bounding Box'))

        dct = {}
        dct['Angle'] = x['Angle']
        dct['Text'] = x['Text']
        dct['orgbox'] = x['orgbox']
        dct['category'] = x['label']
        dct['char_num'] = len(x['Text'].strip())
        dct['bbox'] = json.loads(get_bbox_tokens(temp_bbox, psdsize))
        dct['Bounding Box'] = temp_bbox
        # dct['Bounding Box'] = x['Bounding Box']
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
def serail(img):
    import io
    import base64
    import json
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')

    # 将字节流编码为 Base64
    im_b64 = base64.b64encode(buffer.getvalue()).decode("utf8")
    return json.dumps(im_b64)
def deserail(base64_str):
    from io import BytesIO
    import base64
    from PIL import Image
    byte_data = base64.b64decode(base64_str)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img

def get_bg(bgimg, bgbox, sz):
    a, b, c,d = bgbox
    w, h = sz
    bgimg = np.array(deserail(bgimg))[abs(min(0,b)):, abs(min(0,a)):][:h,:w]
    return bgimg
def bbox_real(bbox,size):
    x0,y0,x1,y1 = bbox
    width, height = size[0],size[1]
    return [x0*width, y0*height, x1*width, y1*height]

def get_bg_canvas(canvas, bgimg, bgbox, sz):
    a, b, c,d = bgbox
    w, h = sz
    bgimg = np.array(deserail(bgimg))[abs(min(0,b)):, abs(min(0,a)):]# [:h,:w]
    mask = bgimg[...,-1:]/255
    h, w = bgimg.shape[:2]
    qi = max(0,b)
    qi2 =  max(0,a)
    ih, iw = canvas[qi:qi+h, qi2:qi2+w].shape[:2]
    canvas[qi:qi+h, qi2:qi2+w] = canvas[qi:qi+h, qi2:qi2+w]*(1-mask[:ih, :iw]) + bgimg[:ih, :iw, :3]*mask[:ih, :iw]
    return canvas
def get_canvas(sz):
    return (np.zeros((sz[1], sz[0], 3))+255).astype(np.uint8)
def get_bg_full(sz, bgbox, bgimg, lst):
    cvs = get_canvas(sz)
    bgimg = get_bg_canvas(cvs, bgimg, bgbox, sz)
    for x in lst:
        if x['Text'] == '':
            # print(x['Text'])
            fg = np.array(deserail(x['img']))
            orgbox = x['orgbox']
            # sz = x['psd_size']
            angle = x['Angle']
            fg_on_canvas, corners = warpPers(fg, orgbox,sz,angle)
            mask = fg_on_canvas[...,-1:]/255
            bgimg = bgimg * (1-mask) + fg_on_canvas[...,:3]*mask
    
    return Image.fromarray(bgimg.astype(np.uint8))

def draw_all_crello(sz, bgimg, lst, flip):
    a, b, c,d = [0,0,sz[0],sz[1]]
    # w, h = sz
    # bgimg = np.array(deserail(bgimg))[abs(min(0,b)):, abs(min(0,a)):][:h,:w]
    h, w = bgimg.shape[:2]
    if flip:
        bgimg = bgimg[:, ::-1]
    if flip:
        for x in lst:
            if x['Text'] == '':
                continue
            fg = np.array(deserail(x['img']))
            orgbox = x['orgbox']
            # sz = x['psd_size']
            angle = x['Angle']
            fg_on_canvas, corners = warpPers(fg, orgbox,sz,angle)
            gua = get_bbox_from_gray(fg_on_canvas[...,-1])
            if gua is not None:
                x0, y0, x1, y1 = gua
            else:
                x['bad'] = True
                continue
            if a > 0:
                x0 = x0-a
            if b >0:
                y0 = y0-b
            # print((w - x0, y0, w - x1, y1))
            x['Bounding Box'] = [w - x1+1, y0, w - x0-1, y1]
            x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], (w, h)))
            try:
                bgimg, rev = put_text_bg_crello(fg_on_canvas[y0:y1, x0:x1], (w - x1+1, y0, w - x0-1, y1),bgimg)
                if rev:
                    x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])
            except Exception as e:
                x['bad'] = True
                traceback.print_exc()
                continue
    else:
        for x in lst:
            if x['Text'] == '':
                continue
            fg = np.array(deserail(x['img']))
            orgbox = x['orgbox']
            # sz = x['psd_size']
            angle = x['Angle']
            fg_on_canvas, corners = warpPers(fg, orgbox,sz,angle)
            gua = get_bbox_from_gray(fg_on_canvas[...,-1])
            if gua is not None:
                x0, y0, x1, y1 = gua
            else:
                x['bad'] = True
                continue
            if a > 0:
                x0 = x0-a
            if b >0:
                y0 = y0-b
            x['Bounding Box'] = [x0, y0, x1, y1]
            x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], (w, h)))
            try:
                bgimg, rev = put_text_bg_crello(fg_on_canvas[y0:y1, x0:x1], (x0, y0, x1, y1),bgimg)
                if rev:
                    x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])
            except:
                x['bad'] = True
                traceback.print_exc()
                continue
    return Image.fromarray(bgimg)

def zb_draw_all_crello(poster_infer_layers, sz, bgimg, lst, flip):
    a, b, c,d = [0,0,sz[0],sz[1]]
    # w, h = sz
    # bgimg = np.array(deserail(bgimg))[abs(min(0,b)):, abs(min(0,a)):][:h,:w]
    h, w = bgimg.shape[:2]

    for idx, x in  enumerate(lst): 
        infer_item = poster_infer_layers[idx]
        new_bbox = infer_item['Bounding Box']
        xx0, yy0, xx1, yy1 = new_bbox

        if x['Text'] == '':
            continue
        fg = np.array(deserail(x['img']))
        # fg = np.array(get_my_poster_png(x['img']))
        orgbox = x['orgbox']
        # sz = x['psd_size']
        angle = x['Angle']
        fg_on_canvas, corners = warpPers(fg, orgbox,sz,angle)
        gua = get_bbox_from_gray(fg_on_canvas[...,-1])
        if gua is not None:
            x0, y0, x1, y1 = gua
        else:
            x['bad'] = True
            continue
        if a > 0:
            x0 = x0-a
        if b >0:
            y0 = y0-b
        x['Bounding Box'] = [x0, y0, x1, y1]
        x['bbox'] = json.loads(get_bbox_tokens_2(x['Bounding Box'], (w, h)))
        # 这里我认为 fg_on_canvas[y0:y1, x0:x1]  裁剪出来的就是好的  我需要resize 并放到infer 里面给出的bbox位置上

        target = cv2.resize(fg_on_canvas[y0:y1, x0:x1], (xx1-xx0, yy1-yy0))

        # try:
        bgimg, rev = put_text_bg_crello(target, (xx0, yy0, xx1, yy1) ,bgimg)
        # bgimg, rev = put_text_bg_crello(fg_on_canvas[y0:y1, x0:x1], (x0, y0, x1, y1),bgimg)
        if rev:
            x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])

    return Image.fromarray(bgimg)

def zb_draw_all_bigposter(poster_infer_layers, sz, bgimg, lst, flip):
    a, b, c,d = [0,0,sz[0],sz[1]]
    # w, h = sz
    # bgimg = np.array(deserail(bgimg))[abs(min(0,b)):, abs(min(0,a)):][:h,:w]
    h, w = bgimg.shape[:2]

    for idx, x in  enumerate(lst): 
        infer_item = poster_infer_layers[idx]
        new_bbox = infer_item['Bounding Box']
        xx0, yy0, xx1, yy1 = new_bbox

        if x['Text'] == '':
            continue
        fg = np.array(get_my_poster_png(x['img']))
        orgbox = x['orgbox']
        # sz = x['psd_size']
        angle = x['Angle']
        fg_on_canvas, corners = warpPers(fg, orgbox,sz,angle)
        gua = get_bbox_from_gray(fg_on_canvas[...,-1])
        if gua is not None:
            x0, y0, x1, y1 = gua
        else:
            x['bad'] = True
            continue
        if a > 0:
            x0 = x0-a
        if b >0:
            y0 = y0-b
        x['Bounding Box'] = [x0, y0, x1, y1]
        x['bbox'] = json.loads(get_bbox_tokens_2(x['Bounding Box'], (w, h)))
        # 这里我认为 fg_on_canvas[y0:y1, x0:x1]  裁剪出来的就是好的  我需要resize 并放到infer 里面给出的bbox位置上

        target = cv2.resize(fg_on_canvas[y0:y1, x0:x1], (xx1-xx0, yy1-yy0))

        # try:
        bgimg, rev = put_text_bg_crello(target, (xx0, yy0, xx1, yy1) ,bgimg)
        # bgimg, rev = put_text_bg_crello(fg_on_canvas[y0:y1, x0:x1], (x0, y0, x1, y1),bgimg)
        if rev:
            x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])

    return Image.fromarray(bgimg)

def zb_draw_all_crello_for_readnoise(poster_infer_layers, sz, bgimg, lst, flip):
    '''
    推理结果poster_infer_layers 中给的是绝对坐标  我用 来替换 从相对坐标转换后的绝对坐标 
    那如果是我的read noise 代码 直接替换 orgbox 就行了
    '''
    a, b, c,d = [0,0,sz[0],sz[1]]
    # w, h = sz
    # bgimg = np.array(deserail(bgimg))[abs(min(0,b)):, abs(min(0,a)):][:h,:w]
    bgimg = np.array(bgimg)
    h, w = bgimg.shape[:2]

    for idx, x in  enumerate(lst): 
        new_bbox = poster_infer_layers[idx]
        xx0, yy0, xx1, yy1 = new_bbox

        if x['Text'] == '':
            continue
        fg = np.array(deserail(x['img']))
        orgbox = x['orgbox']
        # sz = x['psd_size']
        angle = x['Angle']
        fg_on_canvas, corners = warpPers(fg, orgbox,sz,angle)
        gua = get_bbox_from_gray(fg_on_canvas[...,-1])
        if gua is not None:
            x0, y0, x1, y1 = gua
        else:
            x['bad'] = True
            continue
        if a > 0:
            x0 = x0-a
        if b >0:
            y0 = y0-b
        x['Bounding Box'] = [x0, y0, x1, y1]
        x['bbox'] = json.loads(get_bbox_tokens_2(x['Bounding Box'], (w, h)))
        # 这里我认为 fg_on_canvas[y0:y1, x0:x1]  裁剪出来的就是好的  我需要resize 并放到infer 里面给出的bbox位置上

        target = cv2.resize(fg_on_canvas[y0:y1, x0:x1], (xx1-xx0, yy1-yy0))
        
        # try:
        bgimg, rev = put_text_bg_crello(target, (xx0, yy0, xx1, yy1) ,bgimg)
        # bgimg, rev = put_text_bg_crello(fg_on_canvas[y0:y1, x0:x1], (x0, y0, x1, y1),bgimg)
        if rev:
            x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])
        # except:
        #     x['bad'] = True
        #     # traceback.print_exc()
        #     continue
    return Image.fromarray(bgimg)


def color_block_draw_all_crello(sz, bgimg, lst, flip, COLORS):
    # sz 尺寸信息 
    a, b, c,d = [0,0,sz[0],sz[1]]

    h, w = bgimg.shape[:2]

    for x in lst:
        cur_category = x['category']
        cur_color = COLORS[cur_category]
        if x['Text'] == '':
            continue
        fg = np.array(deserail(x['img']))
        orgbox = x['orgbox']
        # sz = x['psd_size']
        angle = x['Angle']
        fg_on_canvas, corners = warpPers(fg, orgbox,sz,angle)
        # 对 fg_on_canvas 所需要的操作  

        gua = get_bbox_from_gray(fg_on_canvas[...,-1])

        cur_array = np.full(fg_on_canvas.shape, cur_color)[...,:3]
        if gua is not None:
            x0, y0, x1, y1 = gua
        else:
            x['bad'] = True
            continue
        if a > 0:
            x0 = x0-a
        if b >0:
            y0 = y0-b
        x['Bounding Box'] = [x0, y0, x1, y1]
        x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], (w, h)))
        # try:
        bgimg, rev = put_text_bg_crello(fg_on_canvas[y0:y1, x0:x1], (x0, y0, x1, y1),bgimg)
        bgimg[y0:y1, x0:x1] = cur_array[y0:y1, x0:x1]
        if rev:
            x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])
        # except:
        #     x['bad'] = True
        #     traceback.print_exc()
        #     continue
    return Image.fromarray(bgimg)

def draw_all_crello_refine(sz, bgimg, lst, flip):
    a, b, c,d = [0,0,sz[0],sz[1]]
    # w, h = sz
    # bgimg = np.array(deserail(bgimg))[abs(min(0,b)):, abs(min(0,a)):][:h,:w]
    h, w = bgimg.shape[:2]
    if flip:
        bgimg = bgimg[:, ::-1]
    if flip:
        for x in lst:
            if x['Text'] == '':
                continue
            x0, y0, x1, y1 = x['Bounding Box']
            x['Bounding Box'] = [w - x1+1, y0, w - x0-1, y1]
            x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], (w, h)))
            fg = np.array(deserail(x['img']))
            orgbox = x['orgbox']
            # sz = x['psd_size']
            angle = x['Angle']
            fg_on_canvas, corners = warpPers(fg, orgbox,sz,angle)
            x0, y0, x1, y1 = get_bbox_from_gray(fg_on_canvas[...,-1])
            if a > 0:
                x0 = x0-a
            if b >0:
                y0 = y0-b
            # print((w - x0, y0, w - x1, y1))
            
            
            try:
                bgimg, rev = put_text_bg_crello(fg_on_canvas[y0:y1, x0:x1], x['Bounding Box'],bgimg)
                if rev:
                    x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])
            except:
                print('pass')
                raise
    else:
        for x in lst:
            if x['Text'] == '':
                continue
            x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], (w, h)))
            fg = np.array(deserail(x['img']))
            orgbox = x['orgbox']
            # sz = x['psd_size']
            angle = x['Angle']
            fg_on_canvas, corners = warpPers(fg, orgbox,sz,angle)
            x0, y0, x1, y1 = get_bbox_from_gray(fg_on_canvas[...,-1])
            if a > 0:
                x0 = x0-a
            if b >0:
                y0 = y0-b

            try:
                bgimg, rev = put_text_bg_crello(fg_on_canvas[y0:y1, x0:x1], x['Bounding Box'],bgimg)
                if rev:
                    x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])
            except:
                print('pass')
                raise
    return Image.fromarray(bgimg)