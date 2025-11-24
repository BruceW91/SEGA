import cv2
from llava.train.val.instruct_tune_simple_text import extract_all
import numpy as np
from PIL import Image 
# from

# 需要一个category 到 颜色的映射:
color_dict = {
    'Bodytext': (255, 0, 0),
    'underlay': (255, 165, 0),
    'Title': (255, 255, 0),
    'Subtitle': (127, 255, 0),
    'Name': (0, 255, 0),
    'Website': (0, 255, 255),
    'Calls to Action': (0, 0, 255),
    'Date': (255, 0, 255),
    'Phone number': (128, 0, 128),
    'Location': (165, 42, 42),
    'Detailed items': (255, 192, 203),
    'Social Media': (128, 128, 128),
    'Menu Items': (255, 255, 255),
    'Others': (0, 0, 0)
}


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

def draw_one(npimg, tp):

    # label = tp[0][:-1]
    # output = ot = process_output(tp[1], npimg.shape[:2][::-1])
    # charnum = tp[2]
    # fontsize = tp[3]
    # color = tp[4]
    # al = tp[5]
    label = tp[0]
    bbox_resolute = tp[1]

    cur_color = color_dict.get(label, (0, 0, 0))

    # text = f"{label}/{charnum}/{fontsize}/{color}/{al}"
    # 之后一直是这个5pixel了
    cv2.rectangle(npimg,tuple(bbox_resolute[:2]),tuple(bbox_resolute[2:]),cur_color,5)
    # cv2.rectangle(npimg,tuple(bbox_resolute[:2]),tuple(bbox_resolute[2:]),cur_color,3)
    # cv2.putText(img=npimg,  org=tuple(output[:2]), \
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=colort,thickness=3)
    return npimg

def visualize_output(output, image):
    '''
    这里的output 是model pred 直接给出的  主要是用其中的bbox 
    但核心是用 process_output 后的绝对坐标
    '''
    # npimg = np.array(rgba2rgb(image))
    for x in output:
        draw_one(image, x)
    return Image.fromarray(image)