import json
from PIL import Image
import numpy as np
from .tools import convert_dct_list, give_order, total_valid, add_newline
from .crello import *

def convert(s, flip):
    layers = convert_dct_list_crello(s['layers'])
    layers = give_order(layers)
    
    bgimg = np.array(get_bg_full(s['layers'][0]['psd_size'], s['bbox'], s['background'], layers))
    inpimg = draw_all_crello((bgimg.shape[1], bgimg.shape[0]), bgimg, layers, flip).convert('RGB')
    lys = [x for x in layers if x['Text']!='' and not x.get('bad', False)]
    layers = lys
    [(x.pop('img'), x.pop('Text'), x.pop("Bounding Box"), x.pop('orgbox'), x.pop('Angle')) for x in layers]
    iper = add_newline(json.dumps([{"category": x["category"], "char_num":x["char_num"]} for x in layers]))
    dialog = [
        {'from': 'human',
   'value': '<image>\n Please detect all text blocks in the image, and output the corresponding category, char_num, bbox, fontsize, fontcolor, alignment for each text block. There are total 13 categories (Title, Subtitle, Bodytext, Date, Name, Website, Phone number, Detailed items, Calls to Action, Menu Items, Social Media, Location, Others), the bbox is formatted as [x_min, y_min, x_max, y_max] representing the bounding box around the text block, char_num refers to the total number of recognizable characters within a bbox, fontsize is a floating point number between 0-1 representing the ratio of the font area to the image area, fontcolor is formatted as [red, green, blue] representing the color of characters in the bbox, alignment determines the position(left, right, center) of texts in the bbox. The answer should start with telling the number of text blocks (e.g. There are xxx text blocks in the image.),  and then output detailed metadata in json format [{"category": xxx, "char_num": xxx, "bbox": [x_min, y_min, x_max, y_max], "fontsize": xxx, "fontcolor": [red, green, blue], "alignment":xxx}, {"category": xxx, "char_num":xxx, "bbox": [x_min, y_min, x_max, y_max], "fontsize": xxx, "fontcolor": [red, green, blue], "alignment":xxx}, ...]'},
        {
            'from': 'gpt',
            'value': f'There are total {len(layers)} text blocks in the image.\n' + add_newline(json.dumps(layers))
        }
    ]
    if flip:
        bgimg = Image.fromarray(np.array(bgimg)[:, ::-1])
    else:
        bgimg = Image.fromarray(np.array(bgimg))
    return dialog, inpimg