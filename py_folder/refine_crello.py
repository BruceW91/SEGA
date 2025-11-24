import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str)
parser.add_argument("--card", type=int)
# parser.add_argument("--base", type=str,default="/data1/zb/LLaMA-Factory-main/sft_merge")
parser.add_argument("--base", type=str,default="/data1/zb/ckpts/big_poster")
# parser.add_argument("--base", type=str,default="/data1/zb/ckpts/llava13b")
parser.add_argument("--lora", type=str)
args = args2 = parser.parse_args()
from pathlib import Path
outp = Path(args.out)
name = outp.name
par = outp.parent
outpkl = par/(name+'pkl')
import sys
sys.path.append("/data1/zb/LLaMA-Factory-main")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.card)
import traceback
import AwesomePoster.skia_utils.refine as srefine
import AwesomePoster.skia_utils.fonts as sfonts
import AwesomePoster.skia_utils.main_utils as smain_utils
sfonts.init('./AwesomePoster/fonts')
from AwesomePoster.control.paste import paste_text_on_bg
# from llava.model.builder import load_pretrained_model_w_vit
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
import pickle as pkl
import json
import sys
import os
from llava.train.val.instruct_tune_simple_text import *

from llava.dataset.subtask import zb_refine_11_1_align_convert, zb_refine_baseline_plusbbox_add_u_allign

def load_model(args):
    disable_torch_init()

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, args.model_name
    )

    args.tokenizer = tokenizer
    args.model = model
    args.image_processor = image_processor
    return args

def generate_bbox(image, prompt, conv_mode, model, image_processor, tokenizer):
    # promptb = prompt.split('<image>\n')[-1].strip()
    # prompt = '<image>\n' + promptb# + 'all boxsize should be very big.'
    image = rgba2rgb(image)
    conv = conv_templates[conv_mode].copy()
    conversations = []
    conv.append_message(conv.roles[0], prompt)
    conversations.append(conv.get_prompt() + ' ASSISTANT:')
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0).to(model.device)
    images_tensor = process_images(
        [image],
        # [image, ex_img],
        image_processor,
        model.config
    ).to(model.device, dtype=model.dtype)
    # print(input_ids)
    # return images_tensor
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():

        output_ids = model.generate(
            inputs=input_ids,
            images=images_tensor,
            do_sample=True,
            temperature=0.2,
            top_p=1.0,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria],
        )
        # output_ids = model.generate(input_ids, images=images_tensor, do_sample=False, temperature=0, top_p=None, num_beams=1, max_new_tokens=512, use_cache=True, stopping_criteria=[stopping_criteria],)
    return output_ids

def extract_all(output):
    
    category_raw = extract_liju_sentences(output, '\"category\"', '\"bbox\"')
    category = []
    for x in category_raw:
        try:
            category.append(x.split(',')[0])
        except:
            category.append('FAIL')
    
    bbox_raw = extract_liju_sentences(output, '\"bbox\"', '}')
    bbox = []
    for x in bbox_raw:
        try:
            x = '[' + extract_text_in_brackets(x)[0] + ']'
            ga = eval(x)
            if (isinstance(ga, list) or isinstance(ga, tuple)) and len(ga) == 4:
                bbox.append(ga)
            else:
                bbox.append('FAIL')
        except:
            bbox.append('FAIL')

    char_num_raw = extract_liju_sentences(output, '\"char_num\"', '\"bbox\"')
    char_num = []
    for x in char_num_raw:
        try:
            char_num.append(json.loads(x.split(',')[0]))
        except:
            char_num.append('FAIL')
    
    fontsize_raw = extract_liju_sentences(output, '\"fontsize\"', '\"fontcolor\"')
    fontsize = []
    for x in fontsize_raw:
        try:
            fontsize.append(json.loads(x.split(',')[0]))
        except:
            fontsize.append('FAIL')


    fontcolor_raw = extract_liju_sentences(output, '\"fontcolor\"', ', \"')
    print(fontcolor_raw)
    fontcolor = []
    for x in fontcolor_raw:
        try:
            fontcolor.append(json.loads(x.split('"')[0]))
        except:
            try:
                fontcolor.append(json.loads(x.split(' ,')[0]))
            except:
                try:
                    fontcolor.append(json.loads(x[:-1]))
                except:
                    fontcolor.append('FAIL')
    # fontcolor[-1] = fontcolor[-1][:3]

    alignment_raw = extract_liju_sentences(output, '\"alignment\"', '}')
    alignment = []
    for x in alignment_raw:
        try:
            alignment.append(x.split(',')[0])
        except:
            alignment.append('FAIL')
    
    category, char_num, fontsize, fontcolor, alignment = [depo(x, bbox) for x in (category, char_num, fontsize, fontcolor, alignment)]
    others = [category, bbox, char_num, fontsize, fontcolor, alignment]
    inder = [i for i, x in enumerate(bbox) if x != 'FAIL']
    for y in others:
        y = [y[x] for x in inder]

    return list(zip(*others))

class dummpy:
    pass

def paint(layers,outputs,background,text_set):

    #background = Image.open(json_dict['image'])
    if len(layers) == len(text_set):
        psd = dummpy()
        psd.size= background.size
        psd.width, psd.height = background.size
        adapted_layers = srefine.adapt_loop_naive(layers,psd.size)
        a_ls, lys = srefine.adapt_loop(copy.deepcopy(adapted_layers), psd.size)
        # final = paste_text_on_bg(psd, a_ls, background)


        list_to_rewrite = []
        for i, x in enumerate(lys):
            ocr_paragraphs = x['OCR_paragraphs']
            lines = []
            for j, line in enumerate(ocr_paragraphs):
                line_list = []
                for k ,string in enumerate(line):
                    line_list.append(string['Text'])
                #line = add_string_to_complete(line, x['Bounding Box'])
                #lys[i]['OCR_paragraphs'][j][-1] = line
                lines.append(line_list)
            list_to_rewrite.append({'full_text': x['Text'],'paragraph': lines})

        # 把改写后的文本写入paragraph
        #llss = put_rewrite_text_into_ori_text_changeline(llss, copy.deepcopy(list_to_rewrite))

        lys = smain_utils.put_rewrite_text_into_ori_layer_changeline(lys, copy.deepcopy(list_to_rewrite), copy.deepcopy(list_to_rewrite))

        llss = lys
        lys, layers_bboxes = smain_utils.get_layer_bbox_after_merging_rewrite(lys)
        
        # 微调
        adapted_layers,_lys = srefine.adapt_loop_2(llss, psd.size) 
        final = paste_text_on_bg(psd, adapted_layers, background)
    else:
        final = np.array(background)    

    
    image = np.array(background)       
    
    for ix, output in enumerate(outputs):
        cv2.rectangle(image,tuple(output[:2]),tuple(output[2:]),(255,0,0),3)
        cv2.putText(img=image, text = str(ix)+'pred', org=tuple(output[:2]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(255,0,0),thickness=3)


    bbox_img = Image.fromarray(image)

    final_img = copy.deepcopy(final)

    final_img = Image.fromarray(final_img) 

    for ix, output in enumerate(outputs):
        cv2.rectangle(final,tuple(output[:2]),tuple(output[2:]),(255,0,0),3)
        cv2.putText(img=final, text = str(ix)+'pred', org=tuple(output[:2]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(255,0,0),thickness=3) 
    
    final_mix = Image.fromarray(final) 

    return final_img, final_mix, bbox_img

def process_justi(justi):
    if justi == 'left':
        return 0
    elif justi == 'right':
        return 1
    else:
        return 2

def process_fontsize(fontsize,size):
    ratio = max(size)
    return fontsize*ratio

def create_layer_from_dict(text, output,size):
    layer = {}
    layer['Text'] = text
    layer['Bounding Box'] = process_output(output['bbox'],size)
    fontsize = 0.5
    if fontsize != 0:
        if fontsize < 1:
            layer['FontSize'] = process_fontsize(fontsize,size)
        else:
            layer['FontSize']  = fontsize 
    else:
        raise ValueError('Font Size = 0')  #process_fontsize(fontsize,size)
    layer['Justification'] = process_justi(output['alignment'])
    layer['FillColor'] = output['fontcolor'][:3]
    layer['Font'] = 'FZFWTongQPOPTJW'
    layer['Tracking'] = 0
    return layer

model_path = args.lora # wjh
model_base = args.base # wjh
conv_mode = 'zb_refine'

args = type('Args', (), {
    "model_path": model_path,
    "model_base": model_base,
    "model_name": get_model_name_from_path(model_path) +'llava_lora',
    "conv_mode": conv_mode,
    "sep": ",",
    "temperature": 0.2,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 4096,
    'json_dict':''
})()

args = load_model(args)

from llava.dataset.subtask.crello_pred_continuev2_underlay import *
from llava.dataset.subtask.tools import *
def serail(img):
    import io
    import base64
    import json
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')

    # 将字节流编码为 Base64
    im_b64 = base64.b64encode(buffer.getvalue()).decode("utf8")
    return json.dumps(im_b64)
def deserail(base64_str):
    from io import BytesIO
    import base64
    from PIL import Image
    byte_data = base64.b64decode(base64_str)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img
def serail(img):
    import io
    import base64
    import json
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')

    # 将字节流编码为 Base64
    im_b64 = base64.b64encode(buffer.getvalue()).decode("utf8")
    return json.dumps(im_b64)
def deserail(base64_str):
    from io import BytesIO
    import base64
    from PIL import Image
    byte_data = base64.b64decode(base64_str)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img

def total_valid(bbox, sz):
    x0,y0,x1,y1 = bbox
    ww, hh = sz
    h, w = y1-y0, x1-x0
    if (w < ww and h < hh) and (x0 >=0 and x1<=ww and y0>=0 and y1<=hh) and (w*h>0):
        return True
    else:
        return False

def genfunc(x, outp, i, pred_p):
    flag=False
    xx= x

    temp_name_wo_pkl = x.name.split('.')[0]
    xx = x = pkl.load(open(x, 'rb'))
    cur_pred = pkl.load(open(pred_p, 'rb'))
    
    for _ in range(1):
        suc = False
        try:
            dialog, image, bgimg = zb_refine_11_1_align_convert.convert_no_score(xx,False,cur_pred)
            
            output_ids = generate_bbox(image, dialog[0]['value'], 'zb_refine', args.model, args.image_processor, args.tokenizer)
            output = args.tokenizer.batch_decode(output_ids)[0]
            
            print(output)
            try:
                outimg = Image.fromarray(visualize_output_bb(output, bgimg, (255,0,0)))
            except:
                outimg = image
                print('画不了')
                continue
            ka = extract_all(output)
            kadct = kadctorg = [dict(zip(('category', 'bbox', 'char_num', 'fontsize', 'fontcolor', 'alignment'), x)) for x in ka]
            layers = []
            outputs = []
            qtxlist = []
            for und in xx['underlay']:
                if total_valid(und, outimg.size):
                # und = [total_valid(x, outimg.size) for x in und]

                    qtxlist.append(
                        {
                            'label':'underlay',
                            'Bounding Box': und
                        }
                    )
            for x, t in zip(kadct, ['a' for x in range(len(kadct))]):
                # print(x)
                bla = create_layer_from_dict(t, x, image.size)
                if x['category'].endswith('"'):
                    bla['label'] = x['category'][:-1]
                else:
                    bla['label'] = x['category']
                qtxlist.append(bla)
                layers.append(bla)
                outputs.append(layers[-1]['Bounding Box'])
            forqtx = {'layers':qtxlist, 'image':serail(bgimg)}
            # final_img, final_mix, bbox_img = paint(layers, outputs, image, ['a' for x in layers])
            outimg.save(outp/f'{temp_name_wo_pkl}.png')
            suc = True
            break
        except:
            print(traceback.print_exc())
    if not suc:
        return []
    return forqtx

outp.mkdir(parents=True, exist_ok=True)
outpkl.mkdir(parents=True, exist_ok=True)
valset = Path('/data1/zb/data/zcrellow_test_underlay/').glob(
    '*.pkl'
)


# pred_p = 'infer_out_refine/refine_11final_model_input_2250pkl'  # 
pred_p = '/data1/zb/LLaMA-Factory-main/temp_out/debugpkl'  # 

valset = list(valset)
valset.sort()

# rag_indexes = torch.load(rag_path)
lst = [x for x in range(len(valset))][int(args2.card)::8]
for i in lst:
    
    # temp_pred = pkl.load(open(temp_pred_p, 'rb'))
    temp_name = valset[i].name
    temp_name_wo_pkl = temp_name.split('.')[0]
    temp_pred_p = pred_p + '/' + temp_name_wo_pkl + '.pkl'
    # example_pk = rag_indexes[temp_name_wo_pkl][0]+'.pkl'
    try:
        res = genfunc(valset[i], outp, i, temp_pred_p)
        if len(res)==0:
            continue
        
        pkl.dump(res, open(outpkl/temp_name, 'wb'))
    except:
        continue