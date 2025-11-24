from llava.eval.run_llava import *
import torch
import torchvision.ops.boxes as bops

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


# class LlaVaProcessor:
#     def __init__(self, tokenizer, image_processor, mm_use_im_start_end):
#         self.mm_use_im_start_end = mm_use_im_start_end
#         self.tokenizer = tokenizer
#         self.image_processor = image_processor
#         self.conv_mode = "llava_v1"

#     def load_demo_images(image_files: Union[List[str], str]):
#         if type(image_files) is list:
#             out = []
#             for image_file in image_files:
#                 image = Image.open(image_file).convert("RGB")
#                 out.append(image)
#         else:
#             out = Image.open(image_files).convert("RGB")
#         return out

#     # TODO: refactor this, not working
#     def get_processed_tokens_demo(self, text: str, image_files: Union[List[str], str]):
#         if self.mm_use_im_start_end:
#             qs = (
#                 qs
#                 + "\n"
#                 + DEFAULT_IM_START_TOKEN
#                 + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
#                 + DEFAULT_IM_END_TOKEN
#                 + "\n"
#                 + DEFAULT_IM_START_TOKEN
#                 + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
#                 + DEFAULT_IM_END_TOKEN
#             )
#         else:
#             qs = (
#                 qs
#                 + "\n"
#                 + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
#                 + "\n"
#                 + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
#             )

#         conv = conv_templates[self.conv_mode].copy()
#         conv.append_message(conv.roles[0], text)
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()

#         images = self.load_demo_images(image_files)
#         image_tensor = torch.stack(
#             [self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
#         )

#         input_ids = (
#             tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
#         )

#         return image_tensor, input_ids

#     def format_text(self, text: str):
#         if self.mm_use_im_start_end:
#             text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
#         else:
#             text = DEFAULT_IMAGE_TOKEN + "\n" + text

#         conv = conv_templates[self.conv_mode].copy()
#         conv.append_message(conv.roles[0], text)
#         conv.append_message(conv.roles[1], None)
#         text = conv.get_prompt()

#         return text

#     def load_image(self, image_path: str):
#         return Image.open(image_path).convert("RGB")

#     @staticmethod
#     def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
#         """Pad a sequence to the desired max length."""
#         if len(sequence) >= max_length:
#             return sequence
#         return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])

#     def get_processed_tokens(self, text: str, image_path: str):
#         prompt = self.format_text(text)
#         image = self.load_image(image_path)

#         input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
#         image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

#         return image_tensor, input_ids

#     def get_processed_tokens_batch(self, batch_text: List[str], image_paths: List[str]):
#         prompt = [self.format_text(text) for text in batch_text]
#         images = [self.load_image(image_path) for image_path in image_paths]

#         batch_input_ids = [
#             tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompt
#         ]

#         # Determine the maximum length of input_ids in the batch
#         max_len = max([len(seq) for seq in batch_input_ids])
#         # Pad each sequence in input_ids to the max_len
#         padded_input_ids = [self.pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in batch_input_ids]
#         batch_input_ids = torch.stack(padded_input_ids)

#         batch_image_tensor = self.image_processor(images, return_tensors="pt")["pixel_values"]

#         return batch_image_tensor, batch_input_ids

def generate_bbox(json,tid,model,tokenizer,test_img=None):
    prompt = [json['conversations'][cid]['value'] for cid in range(2*tid+1)]#"What are the things I should be cautious about when I visit here?"
    image_files = [json['image']]#"../parse_imgs_4/01_letterhead_template_0.png"
    
    qs = prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    # if IMAGE_PLACEHOLDER in qs[0]:
    #     if model.config.mm_use_im_start_end:
    #         qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    #     else:
    #         qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    # else:
    #     if model.config.mm_use_im_start_end:
    #         qs[0] = image_token_se + "\n" + qs[0]
    #     else:
    #         qs[0] = DEFAULT_IMAGE_TOKEN + "\n" + qs[0]

    # if "llama-2" in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # elif 'poster' in model_name.lower():
    #     conv_mode = 'poster'
    # else:
    #     conv_mode = "llava_v0"

    # if args.conv_mode is not None and conv_mode != args.conv_mode:
    #     print(
    #         "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
    #             conv_mode, args.conv_mode, args.conv_mode
    #         )
    #     )
    # else:
    #     args.conv_mode = conv_mode

    conv = conv_templates['poster'].copy()
    qs.append(None)
    
    for cid in range(2*tid+1):
        conv.append_message(conv.roles[cid%2], qs[cid])
    prompt = conv.get_prompt()
    #print(prompt)

    #image_files = image_parser(args)
    if test_img is None:
        images = load_images(image_files)
    else:
        images = [test_img]
    image_processor = model.get_vision_tower().image_processor
    #images[0].show()
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=model.dtype)
    
    #print(images_tensor.shape)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    # print(input_ids)
    # print(input_ids.shape)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        # import ipdb; ipdb.set_trace()
        output_ids = model.generate(
            inputs = input_ids,
            images=images_tensor,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
        # output_ids = model.generate(inputs = input_ids, images=images_tensor, do_sample=False, temperature=0, top_p=None, num_beams=1, max_new_tokens=512, use_cache=True, stopping_criteria=[stopping_criteria],)
        # import ipdb; ipdb.set_trace()
    return output_ids,images


def get_gt_box(jsons,fid,cid):
    gt = jsons[fid]['conversations'][cid*2+1]['value']
    gt = gt.replace('</','').replace('>',' ').split()
    return [int(i) for i in gt]

def remove_chars_before_first_digit(s: str) -> str:
    # Find the index of the first digit
    index = next((i for i, char in enumerate(s) if char.isdigit()), None)
    # If there's no digit, return the original string
    if index is None:
        return s
    # Return the substring from the first digit to the end
    return s[index:]

def process_output(output,size):
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

def cal_iou(output,gt):
    box1 = torch.tensor([output], dtype=torch.float)
    box2 = torch.tensor([gt], dtype=torch.float)
    iou = bops.box_iou(box1, box2)
    return iou