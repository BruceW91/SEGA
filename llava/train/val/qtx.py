from typing import List, Optional
import numpy as np
from PIL import Image
import copy
import cv2
import torch
from llava.constants import IMAGE_TOKEN_INDEX
from ..llava_trainer import LLaVATrainer
from ...conversation import conv_templates, SeparatorStyle
from ...mm_utils import tokenizer_image_token, process_images, KeywordsStoppingCriteria
from ...eval.run_llava import load_images

def process_image(img, out_size=1080):
    img = img#np.array(rgba2rgb(img))
    size = img.shape[:2]
    image = np.zeros([max(size),max(size),3])
    if size[0] > size[1]:
        diff = (size[0]-size[1]) // 2
        image[:,diff:diff+size[1],:] = img
    else:
        diff = (size[1]-size[0]) // 2
        image[diff:diff+size[0],:,:] = img
    return np.array(cv2.resize(image, (out_size,out_size), interpolation=cv2.INTER_NEAREST),np.int32)

def remove_chars_before_first_digit(s):
    index = next((i for i, char in enumerate(s) if char.isdigit()), None)
    if index is None:
        return s
    return s[index:]

def process_output(output, size):
    max_size = max(list(size))
    min_size = min(list(size))
    size_list = list(size)
    diff = (max_size-min_size)//2
    min_id = size_list.index(min_size)
    output = output.replace('<s>','').replace('</s>','').replace('[','').replace(']','').split(',')

    output = [int(float(remove_chars_before_first_digit(i))*max_size) for i in output]
    if min_id == 0: 
        output[0] = output[0]-diff
        output[2] = output[2]-diff
    else:
        output[1] = output[1]-diff
        output[3] = output[3]-diff

    return output
def generate_bbox(version, json_dict, tid, model, tokenizer, image_processor, test_img=None):
    prompt = [json_dict['conversations'][cid]['value'] for cid in range(2*tid+1)]
    image_files = [json_dict['image']]
    
    qs = prompt
    conv = conv_templates[version].copy()
    qs.append(None)
    
    for cid in range(2*tid+1):
        conv.append_message(conv.roles[cid%2], qs[cid])
    prompt = conv.get_prompt()
    prompt += 'ASSISTANT:'
    # print(self.local_rank, tid, prompt)

    if test_img is None:
        images = load_images(image_files)
    else:
        images = [test_img]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=model.dtype)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs = input_ids,
            images=images_tensor,
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    return output_ids,images

def evaluate(
        trainer: LLaVATrainer,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):

        if True:
            images_ = []
            for fid in range(len(trainer.jsons_dicts)):
                json_dict = trainer.jsons_dicts[fid]
                conversations = trainer.jsons_dicts[fid]['conversations']
                name = trainer.jsons_dicts[fid]['image'].split('/')[-1].split('.png')[0]+'.pkl'

                outputs = []
                gt_boxes = []
                
                image = Image.open(trainer.jsons_dicts[fid]['image'])
                size = image.size
                image = np.array(image)
                for cid in range(int(len(conversations)/2)):
                    gt_box = copy.deepcopy(trainer.jsons_dicts[fid]['conversations'][cid*2+1]['value'])
                    gt_box = np.array(np.array(process_output(gt_box, size)), np.int32)
                    try:
                        output_ids, images = generate_bbox(trainer.args.version, json_dict, cid, trainer.model, trainer.tokenizer, trainer.model.get_vision_tower().image_processor)
                        image = np.array(images[0])
                        size = images[0].size
                        output = trainer.tokenizer.batch_decode(output_ids)[0]

                        output_ = output.replace('<s>','').replace('</s>','').replace('[','').replace(']','').split(',')
                        output_ = [float(remove_chars_before_first_digit(i)) for i in output_]
                        output = np.array(np.array(process_output(output, size)), np.int32)
                        json_dict['conversations'][cid*2+1]['value'] = str(output_)

                        outputs.append(output)
                        gt_boxes.append(gt_box)
                    except:
                        outputs.append([0,0,0,0])
                        gt_boxes.append(gt_box)
                        continue

                for ix, (output, gt_box) in enumerate(zip(outputs,gt_boxes)):
                    cv2.rectangle(image,tuple(output[:2]),tuple(output[2:]),(255,0,0),3)
                    cv2.rectangle(image,tuple(gt_box[:2]),tuple(gt_box[2:]),(0,0,255),3)
                    cv2.putText(img=image, text = str(ix)+'pred', org=tuple(output[:2]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(255,0,0),thickness=3)
                    cv2.putText(img=image, text = str(ix)+'GT', org=tuple(gt_box[:2]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0,0,255),thickness=3)
                images_.append(process_image(image))

            images = np.stack(images_,0)
            images = np.swapaxes(images,1,3)
            images = np.swapaxes(images,2,3)

            trainer.tb_writer.add_images("images_steps_"+str(trainer.state.global_step),np.array(images/255),global_step = trainer.state.global_step)
            trainer.tb_writer.flush()
