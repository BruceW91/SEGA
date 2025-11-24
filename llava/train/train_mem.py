# from llava.train.train_from_a100 import train
import sys
sys.path.append("/home/share/huadjyin/home/wanghaoran/wanghaoran/project/SEGA/LLaVA_poster_multi_task/LLaVA-main")
from llava.train.train import train
import os 
os.environ['TRITON_AUTOTUNE'] = '1'  # 强制重新生成
# import os
# os.environ['MASTER_PORT'] = '29501'
#TORCH_DISTRIBUTED_DEFAULT_PORT = int(MASTER_PORT) if MASTER_PORT else 29500 
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' 
if __name__ == "__main__":
    # train(attn_implementation="flash_attention_2")
    train(attn_implementation=None)
    # train(attn_implementation="sdpa")
