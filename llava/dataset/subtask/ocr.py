import json
from .tools import convert_dct_list, give_order, total_valid, add_newline
from .unionfind import CustomUnionFind, do_overlap
from ..utils.download import download_image
from ..utils.imgproc import draw_all


def convert(pk, flip):
    bg = pk[0]
    bgimg = download_image(bg)
    try:
        layers = convert_dct_list(pk)
    except Exception as e:
        for _ in range(20):
            print('pk:', str(pk))
        raise e
    
    layers = [x for x in layers if total_valid(x, bgimg.size)]
    uf = CustomUnionFind(hash_function=lambda x: (json.dumps(x), tuple(x['bbox'])), compare_function=lambda x, y: do_overlap(x[1],y[1]))
    uf.initialize(layers)
    grp = uf.groups()
    layers = [give_order([json.loads(y[0]) for y in x])[0] for x in grp]
    layers = give_order(layers)

    inpimg = draw_all(bgimg, layers, flip)
    [(x.pop('img'), x.pop("Bounding Box")) for x in layers]
    dialog = [
        {'from': 'human',
   'value': '<image>\n Please detect all text blocks in the image, and output the corresponding category, char_num, bbox, fontsize, fontcolor, alignment for each text block. There are total 13 categories (Title, Subtitle, Bodytext, Date, Name, Website, Phone number, Detailed items, Calls to Action, Menu Items, Social Media, Location, Others), the bbox is formatted as [x_min, y_min, x_max, y_max] representing the bounding box around the text block, char_num refers to the total number of recognizable characters within a bbox, fontsize is a floating point number between 0-1 representing the ratio of the font area to the image area, fontcolor is formatted as [red, green, blue] representing the color of characters in the bbox, alignment determines the position(left, right, center) of texts in the bbox. The answer should start with telling the number of text blocks (e.g. There are xxx text blocks in the image.),  and then output detailed metadata in json format [{"category": xxx, "char_num": xxx, "bbox": [x_min, y_min, x_max, y_max], "fontsize": xxx, "fontcolor": [red, green, blue], "alignment":xxx}, {"category": xxx, "char_num":xxx, "bbox": [x_min, y_min, x_max, y_max], "fontsize": xxx, "fontcolor": [red, green, blue], "alignment":xxx}, ...]'},
        {
            'from': 'gpt',
            'value': f'There are total {len(layers)} text blocks in the image.\n' + add_newline(json.dumps(layers))
        }
    ]
    return dialog, inpimg
