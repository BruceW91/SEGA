import cv2
import numpy as np
from PIL import Image
import PIL
# from .download import get_my_poster_png

from llava.dataset.bigposter.offline_utils import get_my_poster_png
import json
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def process_output(output,size):
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

class WJHException(Exception):
    def __init__(self, message):
        self.message = message

def rgba2rgb(png):
    png = png.convert('RGBA')
    background = Image.new('RGBA', png.size, (255, 255, 255))

    alpha_composite = Image.alpha_composite(background, png).convert("RGB")
    return alpha_composite


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

def put_text_bg(imginp, bbox, bg):
    #mx0, my0, mx1, my1 = bbox


    # mx0 = int(mx0)
    # my0 = int(my0)
    # mx1 = int(mx1)
    # my1 = int(my1)

    bg = np.array(bg)
    mx0, my0, mx1, my1 = bbox
    try:
        imgnper = resize_pad(np.array(imginp), (mx1 - mx0, my1 - my0))
    except:
        raise WJHException(f"图片尺寸过大 {str(bbox)}")
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
def draw_all(bgimg, lst, flip):
    # origimg = bgimg
    bgimg = np.array(bgimg)
    h, w = bgimg.shape[:2]
    if flip:
        bgimg = bgimg[:, ::-1]
    if flip:
        for x in lst:
            x0, y0, x1, y1 = x['Bounding Box']
            # print((w - x0, y0, w - x1, y1))
            x['Bounding Box'] = [w - x1, y0, w - x0, y1]
            x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], (w, h)))
            bgimg, rev = put_text_bg(get_my_poster_png(x['img']), (w - x1, y0, w - x0, y1),bgimg)
            if rev:
                x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])
    else:
        for x in lst:
            x0, y0, x1, y1 = x['Bounding Box']
            # print((x0, y0, x1, y1))
            bgimg, rev = put_text_bg(get_my_poster_png(x['img']), (x0, y0, x1, y1),bgimg)
            if rev:
                x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])
    return Image.fromarray(bgimg)

def draw_all_zb_re(bgimg, lst):
    # origimg = bgimg
    bgimg = np.array(bgimg)
    h, w = bgimg.shape[:2]

    for x in lst:
        x0, y0, x1, y1 = x['Bounding Box']
        # print((x0, y0, x1, y1))
        bgimg, rev = put_text_bg(get_my_poster_png(x['img']), (x0, y0, x1, y1),bgimg)
    return Image.fromarray(bgimg)

def draw_all_for_yewu(bgimg, lst, flip):
    origimg = bgimg
    bgimg = np.array(bgimg)
    h, w = bgimg.shape[:2]
    if flip:
        bgimg = bgimg[:, ::-1]
    if flip:
        for x in lst:
            x0, y0, x1, y1 = x['Bounding Box']
            # print((w - x0, y0, w - x1, y1))
            x['Bounding Box'] = [w - x1, y0, w - x0, y1]
            x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], (w, h)))
            bgimg, rev = put_text_bg(origimg, (w - x1, y0, w - x0, y1),bgimg)
            if rev:
                if "fontcolor" in x:
                    x['fontcolor'] = str([255-x for x in x['fontcolor']])
    else:
        for x in lst:
            x0, y0, x1, y1 = x['Bounding Box']
            # print((x0, y0, x1, y1))
            bgimg, rev = put_text_bg(origimg, (x0, y0, x1, y1),bgimg)
            if rev:
                if "fontcolor" in x:
                    x['fontcolor'] = str([255-x for x in x['fontcolor']])
    return Image.fromarray(bgimg)

def draw_all_button(bgimg, lst, flip):
    bgimg = np.array(bgimg)
    h, w = bgimg.shape[:2]
    if flip:
        bgimg = bgimg[:, ::-1]
    if flip:
        for x in lst:
            gua = []
            for but in x['buttons']:
                x0, y0, x1, y1 = but
                gua.append(json.loads(get_bbox_tokens([w - x1, y0, w - x0, y1], (w, h))))
            x['buttons'] = gua

            x0, y0, x1, y1 = x['Bounding Box']
            # print((w - x0, y0, w - x1, y1))
            x['Bounding Box'] = [w - x1, y0, w - x0, y1]
            x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], (w, h)))
            bgimg, rev = put_text_bg(get_my_poster_png(x['img']), (w - x1, y0, w - x0, y1),bgimg)
            if rev:
                x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])
    else:
        for x in lst:
            gua = []
            for but in x['buttons']:
                gua.append(json.loads(get_bbox_tokens(but, (w, h))))
            x['buttons'] = gua

            x0, y0, x1, y1 = x['Bounding Box']
            # print((x0, y0, x1, y1))
            bgimg, rev = put_text_bg(get_my_poster_png(x['img']), (x0, y0, x1, y1),bgimg)
            if rev:
                x['fontcolor'] = str([1-x for x in json.loads(x['fontcolor'])])
    return Image.fromarray(bgimg)

def judge_color(rgb):
    # 计算亮度
    luminance = np.sqrt(0.299*rgb[0]**2 + 0.587*rgb[1]**2 + 0.114*rgb[2]**2)
    
    # 根据亮度判断颜色
    if luminance < 85:   # 这是一个主观的阈值，你可以根据需要调整
        return 'dark'
    elif 85 <= luminance <= 170:
        return 'medium'
    else:
        return 'bright'
    
def draw_all_cgl(bgimg, lst, flip):
    bgimg = np.array(bgimg)
    w,h = lst[0]['psd_size']
    if flip:
        bgimg = bgimg[:, ::-1]
    if flip:
        for x in lst:
            x0, y0, x1, y1 = x['Bounding Box']
            # print((w - x0, y0, w - x1, y1))
            x['Bounding Box'] = [w - x1, y0, w - x0, y1]
            x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], (w, h)))
        # from ..subtask.tools import give_order_cgl
        # lst = give_order_cgl(lst)
    else:
        for x in lst:
            x0, y0, x1, y1 = x['Bounding Box']
            # print((x0, y0, x1, y1))
    return lst, Image.fromarray(bgimg)

def draw_all_cgl2(bgimg, lst, flip, sz):
    bgimg = np.array(bgimg)
    w,h = sz
    if flip:
        bgimg = bgimg[:, ::-1]
    if flip:
        for x in lst:
            x0, y0, x1, y1 = x['Bounding Box']
            # print((w - x0, y0, w - x1, y1))
            x['Bounding Box'] = [w - x1, y0, w - x0, y1]
            x['bbox'] = json.loads(get_bbox_tokens(x['Bounding Box'], (w, h)))
        # from ..subtask.tools import give_order_cgl
        # lst = give_order_cgl(lst)
    else:
        for x in lst:
            x0, y0, x1, y1 = x['Bounding Box']
            # print((x0, y0, x1, y1))
    return lst, Image.fromarray(bgimg)