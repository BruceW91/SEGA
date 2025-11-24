import os
import json
from PIL import Image

def get_my_poster_png(short_path):
    dir_path = '/home/share/huadjyin/home/lishaoshuai/wanghaoran/data/'
    real_p = os.path.join(dir_path, short_path)
    
    pil_image = Image.open(real_p)
    return pil_image
