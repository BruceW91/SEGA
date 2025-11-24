# 先要对所有数据做salinecy2bbox对转换 以及验证转换的正确性
import cv2
import numpy as np
import os
from tqdm import tqdm

def is_contained(bbox1, bbox2):
    """
    检查 bbox1 是否完全包含在 bbox2 内
    bbox 格式: [x1, y1, x2, y2]
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    return x1_1 >= x1_2 and y1_1 >= y1_2 and x2_1 <= x2_2 and y2_1 <= y2_2

def remove_contained_bboxes(bounding_boxes,total_area):
    """
    移除被其他边界框包含的边界框
    """
    # 计算每个边界框的面积
    areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])/total_area for bbox in bounding_boxes]

    # 用一个列表来存储需要保留的边界框索引
    keep_indices = []

    # 遍历每个边界框，检查它是否被其他边界框包含
    for i, bbox in enumerate(bounding_boxes):
        if areas[i] < 0.05:
            print("too small")
            continue

        if all(not is_contained(bbox, other_bbox) or areas[i] > areas[j] for j, other_bbox in enumerate(bounding_boxes) if i != j):
            keep_indices.append(i)

    # 返回未被包含的边界框
    return [bounding_boxes[i] for i in keep_indices]

def saliency2bbox(saliency_map):
        # pass

    # 选择一个阈值，例如 0.5
    saliency_map = saliency_map / 255
    threshold = 0.5
    binary_map = (saliency_map > threshold).astype(np.uint8) * 255

    # 使用 OpenCV 的连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

    # 提取边界框
    bounding_boxes = []
    for i in range(1, num_labels):  # 跳过背景标签（0）
        x, y, width, height, area = stats[i]
        if area > 0:  # 可以根据需要设置最小面积阈值
            bounding_boxes.append([x, y, x + width, y + height])
    return bounding_boxes

# def remove_bad_bboxes(bounding_boxes):
#     return remove_contained_bboxes(bounding_boxes)

# bbox的极大值抑制

test_saliency_map_folder = 'llava/dataset/zb_refine_noise_utils/intersection_02_test'
# test_saliency_map_folder = '/root/paddlejob/workspace/log/code/RALF-main/saliency_out'
origin_folder = "/root/paddlejob/workspace/log/code/total_data/total_bg_imgs/test_bg_imgs"
dst_foler = 'llava/dataset/zb_refine_noise_utils/debug_show2'
if not os.path.exists(dst_foler):
    os.makedirs(dst_foler)


names = os.listdir(origin_folder)
for name in tqdm(names) :
    print(name)
    saliency_map_p = os.path.join(test_saliency_map_folder, name)
    origin_img_p = os.path.join(origin_folder, name)
    dst_p = os.path.join(dst_foler, name)

    # 假设 saliency_map 是你的显著性图，形状为 (height, width)
    origin_img = cv2.imread(origin_img_p,1)
    h,width = origin_img.shape[:2]
    o_copy = origin_img.copy()
    saliency_map = cv2.imread(saliency_map_p, 0)
    total_area = h*width
    bboxes = saliency2bbox(saliency_map)
    real_bboxes = remove_contained_bboxes(bboxes,total_area)
    saliency_map_for_concate = cv2.cvtColor(saliency_map, cv2.COLOR_GRAY2BGR)
    # 在原始图像上绘制边界框
    for box in real_bboxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(origin_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色边框，线宽为2
    #拼接o_copy 和 origin_img
    origin_img = np.concatenate((o_copy, saliency_map_for_concate, origin_img), axis=1)
    cv2.imwrite(dst_p, origin_img)
    