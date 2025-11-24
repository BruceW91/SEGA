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
    map_dict = {1:'text',2:'logo',3:'underlay'}
    # 这里的content是一个字典
    # key 是 图像短名 
    content = list(content.items())[0]
    # key = content.keys()[0]
    key,value_list = content
    layers = [ {"category":map_dict[category],"absolute_bbox":bbox}  for category,bbox in value_list ]
    img_p = os.path.join(img_folder_p, key)
    cur_img = Image.open(img_p)

    for item in layers:
        item['bbox'] = json.loads(get_bbox_tokens_2(item['absolute_bbox'], cur_img.size))
    # layers  json.loads(get_bbox_tokens_2(x['Bounding Box'], psdsize))

    #  以及需要一个类别到文本的转换  下面需要把列表转换成问题
    iper = json.dumps([{"category": x["category"]} for x in layers])
    output = json.dumps([{"category": x["category"], "bbox": x["bbox"]} for x in layers])

    # 还需要转换为相对坐标

    dialog = [
        {'from': 'human',
    'value': '''<image>\n Given a poster background image, please provide a reasonable poster layout.\n'''},
        {
            'from': 'gpt',
            'value': f'''{output}'''
        }
    ]
    return dialog, cur_img