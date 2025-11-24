# 以测试机为例 看图

import os
import numpy as np
import cv2
from tqdm import tqdm

origin_folder = "/root/paddlejob/workspace/log/code/total_data/total_bg_imgs/train_bg_imgs"
# origin_folder = "/root/paddlejob/workspace/log/code/total_data/total_bg_imgs/test_bg_imgs"
# dst_foler = 'llava/dataset/zb_refine_noise_utils/look_01'
dst_foler = 'llava/dataset/zb_refine_noise_utils/intersection_01_train'
if not os.path.exists(dst_foler):
    os.makedirs(dst_foler)

bas_map_p = '/root/paddlejob/workspace/log/code/RALF-main/saliency_out_train_bas'
is_map_p = '/root/paddlejob/workspace/log/code/RALF-main/saliency_out_train'

names_bas = os.listdir(bas_map_p)
names_is = os.listdir(is_map_p)

total_names = [ name for name in names_bas if name in names_is]
for name in tqdm(total_names):
    dst_p = os.path.join(dst_foler, name)

    bas_img = cv2.imread(os.path.join(bas_map_p, name), 0) /255
    is_img = cv2.imread(os.path.join(is_map_p, name), 0) /255

    threshold = 0.1
    bas_img = (bas_img > threshold).astype(np.uint8) 
    is_img = (is_img > threshold).astype(np.uint8) 

    intersection = bas_img * is_img* 255
    max_map = np.maximum(bas_img, is_img)* 255

    intersection = cv2.cvtColor(intersection, cv2.COLOR_GRAY2BGR)
    max_map = cv2.cvtColor(max_map, cv2.COLOR_GRAY2BGR)
    origin_img = cv2.imread(os.path.join(origin_folder, name))

    # origin_img = np.concatenate((intersection, max_map, origin_img), axis=1)
    cv2.imwrite(dst_p, intersection)
    # origin_img = np.concatenate((intersection, max_map, origin_img), axis=1)
    # cv2.imwrite(dst_p, origin_img)

    # 下面两种写法 一个去最大 一个取交集
    # 弄原图 和两种map的拼接  弄成网页 看50个

    # look_map = (np.abs(bas_img - is_img)).astype('uint8') * 255

