# from llava.train.train_from_a100 import train
from llava.train.train_more_encoder import train
import os 
# import os
# os.environ['MASTER_PORT'] = '29501'
#TORCH_DISTRIBUTED_DEFAULT_PORT = int(MASTER_PORT) if MASTER_PORT else 29500 
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' 
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    # train(attn_implementation="sdpa")
