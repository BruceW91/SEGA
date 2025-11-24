import os
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from transformers import AutoProcessor, AutoModel
import torch

def build_vision_tower(vision_tower_cfg, **kwargs):
    print(f'vision_tower_cfg: {vision_tower_cfg}')
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

# def build_vision_projector_zb_siglip():
#     pass
# def build_vision_tower_zb():
#     folder_p = '/root/paddlejob/workspace/log/code/zb/dino_v2'
#     model = AutoModel.from_pretrained(folder_p)
#     processor = AutoProcessor.from_pretrained(folder_p)

#     return model,processor

def build_vision_tower_zb():
    dinov2_vitl14_reg = torch.hub.load('zb_data/dinov2-main', 'dinov2_vitl14_reg',source='local')

    return dinov2_vitl14_reg

if __name__ == '__main__':
    print('test')
    build_vision_tower_zb()
    print("done")

