import os
import numpy as np
from PIL import Image
import pickle as pkl
import random
# 随机产生的错误 可能导致有多种类型的重复 最后还要做个检查呢

# 目前计划的想法 是先对这两个类型add_occulusion，add_underlay_noise生效  
# 然后写一个overlay的检查函数 最终得到目标prompts
import random
import copy

def select_two_random_integers(k):
    """
    从0到k（包括0和k）的范围内随机选择两个不同的整数。

    :param k: 整数范围的上限
    :return: 一个包含两个不同随机整数的元组
    """
    # 确保k是非负数
    if k < 0:
        raise ValueError("k 必须是非负整数")

    # 随机选择第一个整数
    first_integer = random.randint(0, k)
    
    # 选择第二个整数，确保它与第一个整数不同
    second_integer = random.randint(0, k)
    while second_integer == first_integer:
        second_integer = random.randint(0, k)

    return (first_integer, second_integer)

def copy_perturb(bbox,max_perturbation=0.1):
    # 在给定的bbox基础上 扰动生成一个新的bbox

    x1, y1, x2, y2 = bbox
    
    # 计算宽度和高度
    width = x2 - x1
    height = y2 - y1
    
    # 生成随机扰动量
    dx1 = random.uniform(-max_perturbation, max_perturbation) * width
    dy1 = random.uniform(-max_perturbation, max_perturbation) * height
    dx2 = random.uniform(-max_perturbation, max_perturbation) * width
    dy2 = random.uniform(-max_perturbation, max_perturbation) * height

    # 计算新的边界框坐标
    new_x1 = x1 + dx1
    new_y1 = y1 + dy1
    new_x2 = x2 + dx2
    new_y2 = y2 + dy2
    
    # 确保新的边界框是有效的
    if new_x1 > new_x2:
        new_x1, new_x2 = new_x2, new_x1
    if new_y1 > new_y2:
        new_y1, new_y2 = new_y2, new_y1
    
    # 防止越界
    new_x1 = max(0.0, min(new_x1, 1.0))
    new_y1 = max(0.0, min(new_y1, 1.0))
    new_x2 = max(0.0, min(new_x2, 1.0))
    new_y2 = max(0.0, min(new_y2, 1.0))
    
    return [np.around(new_x1,2), np.around(new_y1,2), np.around(new_x2,2), np.around(new_y2,2)]

def iou(bbox1, bbox2):
    """
    计算两个 BBox 之间的 IoU。
    
    :param bbox1: 第一个 BBox (x1, y1, x2, y2)
    :param bbox2: 第二个 BBox (x1, y1, x2, y2)
    :return: IoU 值
    """
    # 计算交集
    inter_x1 = max(bbox1[0], bbox2[0])
    inter_y1 = max(bbox1[1], bbox2[1])
    inter_x2 = min(bbox1[2], bbox2[2])
    inter_y2 = min(bbox1[3], bbox2[3])

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0  # 没有交集

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # 计算并集
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - 2 * inter_area

    # 计算 IoU
    iou_value = inter_area / (union_area + 1e-6)
    return iou_value

def find_max_iou_bbox(bbox, bboxlist):
    """
    在 bboxlist 中找到与给定 BBox 重叠最多的 BBox。
    
    :param bbox: 给定的 BBox (x1, y1, x2, y2)
    :param bboxlist: BBox 列表 [(x1, y1, x2, y2), ...]
    :return: 重叠最多的 BBox 及其 IoU 值
    """
    max_iou = 0.0
    max_iou_bbox = None
    max_idx = -1

    for idx,candidate in enumerate(bboxlist):
        current_iou = iou(bbox, candidate['bbox'])
        if current_iou > max_iou:
            max_iou = current_iou
            max_iou_bbox = candidate
            max_idx = idx

    return max_idx, max_iou

# 对于我的 盖住saliency map的目的 需要我的 xy 有较大程度的随机性
def perturb_bbox(bbox, max_perturbation=0.1):
    """
    对 (x1, y1, x2, y2) 格式的相对坐标边界框进行扰动，并保证新边界框的有效性。
    
    :param bbox: 相对坐标表示的边界框 (x1, y1, x2, y2)
    :param max_perturbation: 最大扰动量（相对于原值的比例）
    :return: 扰动后的边界框
    """
    x1, y1, x2, y2 = bbox
    
    # 计算宽度和高度
    width = x2 - x1
    height = y2 - y1
    
    # 生成随机扰动量
    new_x1 = random.uniform(0, 1-width) 
    new_y1 = random.uniform(0, 1-height) 

    d_width = random.uniform(-max_perturbation, max_perturbation) * width
    d_height = random.uniform(-max_perturbation, max_perturbation) * height

    width += d_width
    height += d_height
    
    # 计算新的边界框坐标
    new_x2 = new_x1 + width
    new_y2 = new_y1 + height
    
    # 确保新的边界框是有效的
    if new_x1 > new_x2:
        new_x1, new_x2 = new_x2, new_x1
    if new_y1 > new_y2:
        new_y1, new_y2 = new_y2, new_y1
    
    # 防止越界
    new_x1 = max(0.0, min(new_x1, 1.0))
    new_y1 = max(0.0, min(new_y1, 1.0))
    new_x2 = max(0.0, min(new_x2, 1.0))
    new_y2 = max(0.0, min(new_y2, 1.0))
    
    return [np.around(new_x1,2), np.around(new_y1,2), np.around(new_x2,2), np.around(new_y2,2)] 

import numpy as np

def bbox_saliency_overlap(saliency_map, bbox):
    """
    计算给定BBox与0-1的saliency map的重叠占比。
    
    :param saliency_map: 二维numpy数组，表示显著性图，取值范围在[0, 1]之间
    :param bbox: 元组 (x1, y1, x2, y2)，表示相对坐标的BBox
    :return: BBox内显著性图的重叠占比
    """
    # 获取显著性图的高度和宽度
    height, width = saliency_map.shape
    
    # 将相对坐标转换为绝对坐标
    x1, y1, x2, y2 = int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)
    
    # 确保坐标不会越界
    x1, x2 = max(0, min(x1, width-1)), max(0, min(x2, width-1))
    y1, y2 = max(0, min(y1, height-1)), max(0, min(y2, height-1))
    
    # 提取BBox区域内的显著性图
    bbox_region = saliency_map[y1:y2, x1:x2]/255

    total_area = (x2 - x1) * (y2 - y1)
    
    # 计算重叠占比
    overlap_ratio = np.sum(bbox_region) / (total_area+1e-6)
    
    return overlap_ratio


def add_occulusion(layouts,saliency_map):
    # 这边我准备好了 每个poster的saliency_bbox  随机扰动的时候 我只需要在这个范围内生成一个
    # 这个就不能用bbox  解决不了空心的问题 ！！！ 
    # 随机扰动 并check_occulusion 给出有问题的bbox

    # 这里的map 我要用自己的低阈值交集map
    error_list = [ ]
    layouts_c = copy.deepcopy(layouts)

    # 这里获取saliency_map的mask面积 < 0.05 就不弄了
    h,w = saliency_map.shape
    mask_area = np.sum(saliency_map) / 255
    mask_ratio = mask_area / (h*w)
    if mask_ratio < 0.05:
        return layouts,error_list


    cur_lens = len(layouts)
    move_nums = [3,2,1,0]
    probabilities = [0.01,0.04,0.45,0.5]
    selected_value = random.choices(move_nums, weights=probabilities, k=1)[0]
    # 如果层数不超过3 最多坏一个
    selected_value = min(selected_value, cur_lens)

    select_layers = random.sample(range(cur_lens), selected_value)
    for cur_layer in select_layers:
        cur_item = layouts[cur_layer]
        cur_bbox = layouts[cur_layer]['bbox']
        while_num = 0
        # 随机扰动 直到occulusion 满足要求
        while(while_num<30):
            while_num += 1
            # 所有纯while 要有循环退出机制 30遍吧
            new_bbox = perturb_bbox(cur_bbox)
            overlap_ratio = bbox_saliency_overlap(saliency_map, new_bbox)
            if overlap_ratio > 0.3:
                layouts_c.remove(cur_item)
                cur_item['bbox'] = new_bbox
                error_list.append(cur_item)
                
                break

    return layouts_c, error_list


def add_underlay_noise_for_pku(layouts):

    # 原本的逻辑是 针对每个underlay 找到其上面的bbox 然后把他扰动出去
    # 返回没其他正确的layouts 
    print("jjj")
    underlay_bboxs = [ item for item in layouts if item['category'] == 'underlay']
    others = [ item for item in layouts if item['category'] != 'underlay']

    cur_lens = len(underlay_bboxs)
    if len(underlay_bboxs) <= 0 or len(layouts)==0:
        return layouts,[],[]
    pass


def add_underlay_noise(layouts,underlay_bboxs):

    error_list = [ ]
    err_underlay_bboxs = []

    # 找到在underlay_bboxs中的layouts 然后随机扰动
    cur_lens = len(underlay_bboxs)
    if len(underlay_bboxs) <= 0 or len(layouts)==0:
        return layouts,[],[]
    #没得扰动
    else:
        move_nums = [3,2,1,0]
        probabilities = [0.05,0.15,0.4,0.4]
        selected_value = random.choices(move_nums, weights=probabilities, k=1)[0]
        # 如果层数不超过3 最多坏一个
        selected_value = min(selected_value, cur_lens)

        select_layers = random.sample(range(cur_lens), selected_value)
        # 这边选择的是对哪个underlay_bboxs进行扰动
        for cur_layer in select_layers:
            if len(layouts) == 0:
                break
            cur_underlay_bbox = underlay_bboxs[cur_layer]
            try:
                idx,max_iou = find_max_iou_bbox(cur_underlay_bbox, layouts)
                # 这里有可能=-1 就跳过这个underlay
            except:
                print("hh")
            cur_item = layouts[idx]
            
            # cur_item = layouts.pop(int(idx))
            if max_iou > 0.3:
                # 当前bbox是有效的
                # cur_bbox = layouts[idx]['bbox']
                layouts.remove(cur_item)
                cur_bbox = cur_item['bbox']
                new_bbox = perturb_bbox(cur_bbox)
                cur_item['bbox'] = new_bbox
                error_list.append(cur_item)
                err_underlay_bboxs.append(cur_underlay_bbox)
                # layouts[idx]['bbox'] = new_bbox
            else:
                continue

        return layouts, error_list, err_underlay_bboxs



def check_overlay(layouts):

    overlapping_pairs = []

    # 遍历每一对 BBox
    for i in range(len(layouts)):
        for j in range(i + 1, len(layouts)):
            if iou(layouts[i]['bbox'], layouts[j]['bbox'])>0.05:
                overlapping_pairs.append((layouts[i]['bbox'], layouts[j]['bbox']))
    return overlapping_pairs
    # pass

def add_overlay_noise(layouts):
    # 如果当前layouts<2 就跳出循环
    # 之后随机选择一个 bbox 让其消失 对另一个 进行copy扰动, 把这俩移除 当前layouts 
    # 0.8的概率跳出 

    err_bbox_list = []
    bad_list = []
    while(True):

        num_boxes = len(layouts)
        if num_boxes < 3:
            break
        
        # 随机选择两个 BBox 进行扰动  这里用索引 感觉有危险 ！！！
        idx_list = select_two_random_integers(num_boxes-1)
        idx1,idx2  = idx_list 
        item1 = layouts[idx1]
        item2 = layouts[idx2]
        
        # 扰动 BBox 这里默认一定有重叠
        try:
            perturbed_bbox = copy_perturb(item2['bbox'])
        except Exception as e:
            print('Error:', e)
            continue

        # pop原始 BBox 和扰动后的 BBox
        layouts = [ item for idx ,item in enumerate(layouts) if idx not in idx_list ]
        
        err_bbox_list.append((item2['bbox'], perturbed_bbox))

        item1['bbox'] = perturbed_bbox
        bad_list.append(item1)
        bad_list.append(item2)

        # 判断是否满足条件
        if random.random() < 0.8:
            break

    return layouts, bad_list ,err_bbox_list

def add_noise_for_pku(layouts, temp_saliency_map):

    
    non_underlay_layers = [ item for item in layouts if item['category'] != 'underlay']
    underlay_layers = [ item for item in layouts if item['category'] == 'underlay']


    # return total_add_noise_pku(layouts, temp_saliency_map

def total_add_noise_pku(layouts, temp_saliency_map):
    origin_len = len(layouts)

    # origin_layouts = layouts.copy()
    # 每一个加噪声函数 应同时给出一个error_list
    layouts, occlusion_errors = add_occulusion(layouts, temp_saliency_map)

    layouts, underlay_errors, err_underlay = add_underlay_noise_for_pku(layouts)
    # 这个应该是指出underly 上面没有text 

    layouts ,bad_list ,err_overlay  = add_overlay_noise(layouts)

    overlay_errors_plus = check_overlay(layouts) # 这个不改变layouts
    overlay_errors = overlay_errors_plus + err_overlay

    # 错误细节应该都只放bbox
    total_layouts = layouts + occlusion_errors + underlay_errors + bad_list

    if len(total_layouts) != origin_len:
        raise ValueError("The number of elements in the layout list has changed after adding noise.")
    

    # total_layouts这边需要的其实是 所有layouts 带着字数 的 以及错误的
    return total_layouts, occlusion_errors, err_underlay, overlay_errors

if __name__ == '__main__':
    # 示例边界框
    bbox = (0.1, 0.2, 0.7, 0.8)  # 左上角 (0.1, 0.2)，右下角 (0.7, 0.8)

    # 进行扰动
    perturbed_bbox = perturb_bbox(bbox, max_perturbation=0.1)

    print("Original BBox:", bbox)
    print("Perturbed BBox:", perturbed_bbox)