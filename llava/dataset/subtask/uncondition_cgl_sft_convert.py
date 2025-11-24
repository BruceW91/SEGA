import json
from PIL import Image
import numpy as np

import os
from llava.dataset.subtask.tools import convert_dct_list, give_order, total_valid, add_newline

# from .unionfind import CustomUnionFind, do_overlap
# from .tools import convert_dct_list, give_order, total_valid, add_newline, intersection_area
from llava.dataset.subtask.crello import *

def give_order_udl(pd):
    pd = sorted(pd, key=lambda x:x[:2][::-1])
    return pd
def convert(content,img_folder_p):
    map_dict = {1:'Logo',2:'Text',3:'underlay',4:'embellishment',5:'highlighted text'}
    # 这边的img_folder_p 要更换一下！！！

    filename,layers = content
    filename = filename.replace(".jpg",".png")

    img_p = os.path.join(img_folder_p, filename)
    cur_img = Image.open(img_p)

    for item in layers:
        item['bbox_r'] = json.loads(get_bbox_tokens_2(item['bbox'], cur_img.size))

    #  以及需要一个类别到文本的转换  下面需要把列表转换成问题
    iper = json.dumps([{"category":map_dict[x["category_id"]] } for x in layers])
    output = json.dumps([{"category": map_dict[x["category_id"]], "bbox": x["bbox_r"]} for x in layers])

    # 还需要转换为相对坐标

    dialog = [
        {'from': 'human',
    'value': '''<image>\n Given a poster image, please give a reasonable poster layout.\n'''},
        {
            'from': 'gpt',
            'value': f'''{output}'''
        }
    ]
    return dialog, cur_img