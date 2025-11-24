from torch.utils.data import Dataset
import transformers
from scripts.bos.bos_client import get_url

import json
import copy
import numpy as np
import requests
from PIL import Image
from llava.dataset.utils.imgproc import  rgba2rgb
import torch
import PIL.Image
import pickle as pkl
from llava.dataset.preproc import preprocess_multimodal, preprocess_poster
PIL.Image.MAX_IMAGE_PIXELS = 933120000
local_rank = None


from torch.utils.data import DataLoader
# import 子任务
from llava.dataset.subtask import crello_ocrv2_drop, crello_pred_simple, crello_pred_continuev2_underlay_tuili, crello_pred_raw_continuev2_underlay_tuili,zb_online_mixdpo_crello_pred_continuev2_underlay_tuili

def select_task(pk):
    s = pkl.load(open(pk, 'rb'))
    flip = np.random.rand() > 0.5
    return np.random.choice((zb_online_mixdpo_crello_pred_continuev2_underlay_tuili.convert, ))(s, flip)



def rank0_print(*args):
    if local_rank == 0:
        print(*args)
from pathlib import Path
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = sorted(list(Path(data_path).glob('*.pkl')))
        # list_data_dict = list_data_dict[:int(len(list_data_dict)*0.9)]

        # bad_index_list = [1024, 1025, 1026, 1027, 1536, 1029, 1030, 1031, 1541, 1033, 1034, 1035, 1036, 1037, 511, 8, 1544, 530, 531, 1043, 1044, 1046, 1047, 1048, 1049, 1042, 1051, 22, 534, 1054, 1055, 1056, 542, 1058, 546, 36, 1061, 1062, 37, 1064, 41, 1066, 40, 555, 103, 48, 1073, 1074, 1077, 54, 566, 1593, 1083, 1084, 573, 1085, 1088, 1089, 1090, 1091, 1604, 1093, 1094, 577, 1096, 1097, 71, 1099, 1100, 77, 1102, 1103, 591, 1108, 1109, 1110, 600, 1113, 1115, 1116, 1117, 1118, 1119, 1630, 1123, 1124, 613, 1126, 1127, 1128, 1129, 1642, 1130, 620, 104, 618, 107, 109, 1648, 116, 1142, 121, 123, 124, 637, 636, 1151, 1152, 638, 644, 1159, 648, 1045, 139, 1168, 145, 1169, 661, 1175, 664, 667, 668, 156, 1182, 673, 674, 165, 1189, 1702, 1533, 172, 543, 1207, 1211, 1724, 1213, 192, 1731, 710, 716, 1741, 720, 211, 1748, 723, 733, 736, 739, 1251, 232, 236, 751, 1270, 1785, 249, 251, 1277, 768, 260, 1286, 1287, 781, 783, 787, 275, 1308, 1824, 1315, 293, 1318, 294, 1832, 297, 1322, 1323, 808, 1325, 302, 807, 1485, 817, 468, 1845, 309, 469, 470, 1345, 471, 324, 836, 838, 472, 330, 331, 333, 1871, 1361, 856, 1374, 351, 1377, 355, 1382, 1391, 370, 371, 1400, 1601, 1404, 899, 911, 915, 920, 413, 414, 417, 1443, 934, 935, 424, 940, 1454, 431, 947, 436, 437, 1462, 1466, 448, 966, 968, 457, 458, 459, 972, 460, 974, 975, 976, 461, 462, 979, 980, 463, 982, 465, 466, 985, 467, 987, 988, 477, 1502, 990, 991, 993, 1503, 996, 1509, 998, 999, 1510, 1511, 1002, 997, 1004, 1006, 1007, 1008, 1009, 495, 1015, 496, 1013, 1526, 1014, 1523, 1529, 1018, 1011, 1021, 1022, 1023]
        # list_data_dict = [list_data_dict[x] for x in range(len(list_data_dict)) if x not in bad_index_list]
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        valset = list(range(len(list_data_dict)))
        valset = valset[::len(valset)//5][:4]
        self.train_dict = [list_data_dict[x] for x in range(len(list_data_dict)) if x not in valset]
        self.valset = [list_data_dict[x] for x in valset]
        self.data_args = data_args
        rank0_print("Init dataset done")

    def __len__(self):
        return len(self.train_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.train_dict:
            img_tokens = 128#  if 'image' in sample else 0
            length_list.append(5 + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample  in self.train_dict:
            img_tokens = 128#  if 'image' in sample else 0
            length_list.append(5 + img_tokens)
        return length_list

    def __getitem__(self, i):
        while True:
            try:
                sources = self.train_dict[i]
                dialog, nega_dialog, image, done = select_task(sources)
                sources = preprocess_multimodal(
                    [copy.deepcopy(dialog),],
                    self.data_args)
                
                data_dict_chosen = preprocess_poster(
                    sources,
                    self.tokenizer,
                    has_image=True)

                sources_rejected = preprocess_multimodal(
                    [copy.deepcopy(nega_dialog),],
                    self.data_args)

                data_dict_rejected = preprocess_poster(
                    sources_rejected,
                    self.tokenizer,
                    has_image=True )                

                processor = self.data_args.image_processor
                image = rgba2rgb(image)

                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]


                data_dict = dict(input_ids_chosen=data_dict_chosen["input_ids"][0],
                            labels_chosen=data_dict_chosen["labels"][0],
                            input_ids_rejected=data_dict_rejected["input_ids"][0],    #这里用同一个输入input——ids
                            labels_rejected=data_dict_rejected["labels"][0],data_index = i, done_layer = done)

                data_dict['image'] = image
                return data_dict
            except Exception as e:
                print(e)
                # raise
                i = np.random.randint(len(self.train_dict))

if __name__ == '__main__':
    dset = LazySupervisedDataset('/root/paddlejob/workspace/log/code/dataset/zcrellow_train_underlay', None, None)
    dataloader = DataLoader(
    dataset=dset,      # 数据集
    batch_size=10,         # 批量大小
    shuffle=True,          # 是否打乱数据
    num_workers=0,         # 用于数据加载的子进程数，0表示在主进程中加载数据）
    )
    for i, data in enumerate(dataloader):
        print(i)