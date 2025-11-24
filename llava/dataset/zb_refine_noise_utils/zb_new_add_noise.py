import os
from llava.dataset.zb_refine_noise_utils.metric import *
from llava.dataset.zb_refine_noise_utils.util import collate_fn
import copy
import random

LABELS = ['Title', 'Name','Social Media','Website','Phone number','Detailed items','FAIL','Menu Items','Date','Subtitle','Bodytext','Calls to Action','Location','Others', 'Underlay']

def perturb_bbox(item, imginfo ,max_perturbation=0.1):

    """

    这里要全换成整数的
    对 (x1, y1, x2, y2) 格式的相对坐标边界框进行扰动，并保证新边界框的有效性。
    
    :param bbox: 相对坐标表示的边界框 (x1, y1, x2, y2)
    :param max_perturbation: 最大扰动量（相对于原值的比例）
    :return: 扰动后的边界框
    """
    bbox = item['bbox']
    x1, y1, x2, y2 = bbox

    w_pic, h_pic = imginfo
    
    # 计算宽度和高度
    width = x2 - x1
    height = y2 - y1

    while(True):
    
        # 生成随机扰动量
        new_x1 = random.uniform(0, w_pic-width) 
        new_y1 = random.uniform(0, h_pic-height) 

        # new_x1 = random.randint(0, 10)

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
        new_x1 = max(0, min(new_x1, w_pic))
        new_y1 = max(0, min(new_y1, h_pic))
        new_x2 = max(0, min(new_x2, w_pic))
        new_y2 = max(0, min(new_y2, h_pic))
        
        if  new_x2-new_x1 > 0.01 * width and new_y2-new_y1 > 0.01 * height:
            break

    new_bbox = [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
    item['bbox'] = new_bbox
    
    # return [np.around(new_x1,2), np.around(new_y1,2), np.around(new_x2,2), np.around(new_y2,2)] 
    return item

class ClassLabel:
    pass    
def str2int(text):
    return LABELS.index(text)

def calculate_scores(batch):

    feature_label = ClassLabel()
    feature_label.names = ['Title' for i in range(len(LABELS)-1)] + ['underlay']
    feature_label.id = None
    feature_label.num_classes = len(LABELS)
    feature_label.str2int = str2int


    ali_score = compute_alignment(batch)
    over_score = compute_overlay_zb(batch,feature_label)
    
    saliency_scores = compute_saliency_aware_metrics(batch, feature_label)
    underlay_scores = compute_underlay_effectiveness_zb(batch, feature_label)
    return { **ali_score,**over_score ,**saliency_scores,**underlay_scores } 


def get_single_meta(output_list,sizeinfo):

    '''
    因为我其实不需要给出detail  所以这个layouts 和 saliency map 我都还可以用绝对坐标来表示
    '''
    # output_list = result['layers']#[{'category':r[0],'bbox':r[1]} for r in result]

    center_cx = []
    center_cy = []
    widths = []
    heights = []
    categories = []

    width,height = sizeinfo
    
    for ix, output in enumerate(output_list):
        #pred_bbox = process_output(str(output['bbox']),image.size)
        try:
            pred_bbox = output['absolute_bbox']
        except:
            pred_bbox = output['bbox']
        w, h = (pred_bbox[2]-pred_bbox[0])/width, (pred_bbox[3] - pred_bbox[1])/height
        cx, cy = (pred_bbox[0] + w*width//2)/width, (pred_bbox[1] + h*height//2)/height
        center_cx.append(cx)
        center_cy.append(cy)
        widths.append(w)
        heights.append(h)
        try:
            category = output['label']
        except:
            category = output['category']

        if category == 'underlay':
            category = 'Underlay'

        if category != 'Underlay':
            category = 'Title'
        c = LABELS.index(category)
        categories.append(c)

    _result = {}
    _result['center_x'] = center_cx
    _result['center_y'] = center_cy
    _result['width'] = widths
    _result['height'] = heights
    _result['label'] = categories
    # _result['image'] = example_img
    return _result


def get_scores_for_cur_layouts(layouts,saliency_map, cur_img):
    # 这里layouts 传进来绝对的

    # 这里我需要准备好 saliency map   以及对layouts 转换为 需要的格式  这里可以现场弄
    # 先转成字典 再collate
    sizeinfo = saliency_map.size
    layouts_dict = get_single_meta(layouts,sizeinfo)
    layouts_dict['saliency'] = torch.tensor(np.array(saliency_map)/255).unsqueeze(0).float()
    layouts_dict['image'] = torch.tensor(np.array(cur_img)/255).permute(2, 0, 1).float()
    layouts_dict = [layouts_dict]
    valid_score = zb_compute_validity(layouts_dict)
    
    b_layouts = collate_fn(layouts_dict)

    scores_list = calculate_scores(b_layouts)
    scores_list['valid_score'] = valid_score[2]
    return scores_list

def add_noise_and_get_score(layouts, saliency_map, cur_img):

    origin_layers = copy.deepcopy(layouts)
    # while 循环 随机扰动几个框 
    total_num = len(layouts)

    while(True):
        rand_idx = np.random.choice(total_num, 1)[0]
        cur_item = layouts[rand_idx]
        new_item = perturb_bbox(cur_item, cur_img.size)
        layouts[rand_idx] = new_item

        cur_p = random.uniform(0, 1) 
        if cur_p < 0.7 :
            continue
        else:
            break

    # 或许当前布局的先验分数, 下面再根据这个分数 卡阈值 来给出cot句子
    scores_list = get_scores_for_cur_layouts(layouts, saliency_map, cur_img)

    return layouts, scores_list

# add_noise_and_get_score_for_crello
def add_noise_and_get_score_for_crello(layouts, underlays, saliency_map, cur_img):

    origin_layers = copy.deepcopy(layouts)
    # while 循环 随机扰动几个框 
    total_num = len(layouts)

    while(True):
        rand_idx = np.random.choice(total_num, 1)[0]
        cur_item = layouts[rand_idx]
        new_item = perturb_bbox(cur_item, cur_img.size)
        layouts[rand_idx] = new_item

        cur_p = random.uniform(0, 1) 
        if cur_p < 0.6 :
            continue
        else:
            break
    if len(underlays)>0:
        underlay_layouts = [ {'category':'underlay', 'bbox':bbox,'Bounding Box':bbox} for bbox in underlays ]
        total_layouts =  layouts + underlay_layouts
    else:
        total_layouts =  layouts

    # 这边主要是需要layouts 绝对坐标来算分

    # 或许当前布局的先验分数, 下面再根据这个分数 卡阈值 来给出cot句子
    scores_list = get_scores_for_cur_layouts(total_layouts, saliency_map, cur_img)

    return total_layouts,scores_list

    