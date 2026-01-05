# SEGA
**<center><font size=4>[ICCV 2025] SEGA: A Stepwise Evolution Paradigm for Content-Aware Layout Generation
with Design Prior</font></center>**  

[![Conference](https://img.shields.io/badge/ICCV-2025-green)]()
[![Paper](https://img.shields.io/badge/arXiv-2412.10028-brightgreen)](http://arxiv.org/abs/2510.15749)
[![Project](https://img.shields.io/badge/Project-red)](https://brucew91.github.io/SEGA.github.io)


## Updates
- [11/24] We release the inference code and models.
- [10/25] We release the [GenPoster-100K dataset](GenPoster-100K_Dataset).
- [06/25] SEGA is accepted by ICCV 2025.

## Code

### Envs
- cuda_12.x   
- python=3.10
- pytorch 1.7.1
- pip install -r requirements.txt
- python -m spacy download zh_core_web_sm

### Dataset
We provide the PKU-style Crello dataset for quick training and inference.

Link: https://pan.baidu.com/s/1hAyl_LatXIW-PEXnXzRCcw password: 1234 


### Models
We upload all Crello related Models by Baidu Netdisk. (You only need part of them. Below is specific path.)

Link: https://pan.baidu.com/s/1jW7jMjWEOWCgSTU-jUjsNw password: 1234 

Below, if no specific notation, all are Lora folder.

#### 7B Model
- SFT model : zzz_git/sft
- Refiner from Llava :  zzz_git/refine
<!-- - SFT model merged : -->


#### 13B Model

- SFT merged: base_sft.tar
- SFT merged pretrained: big.tar
- SFT model : crello_series1/simple_sft_17e_nocot_all_old
- Refiner from Bigposter : crello_series2/refine_final_11_1_10e_all_from_bigposter (--base use SFT merged pretrained)
- Refiner from SFT : crello_series2/refine_final_11_1_10e_from_sft (--base use SFT merged)
- Refiner from Llava : crello_series2/ refine_final_11_1_10e_basedata_nocot400_70p



### Inference 

#### Preparetion
Download openai/clip-vit-large-patch14-336 in SEGA dir.

Download llava-1.5 7B and 13B for use load checkpoints.

Download Awesomeposter for utils and fonts by link and place it in SEGA dir : https://pan.baidu.com/s/1XoTwYEPbW3rsoKN-VtxOtw 提取码: 1234 

sfonts.init('./AwesomePoster/fonts') ( Focus this path in inference script)

<!-- sys.path.append("/data1/zb/LLaMA-Factory-main") -->

#### Use

- py_folder/zzz_infer_crello.py : SFT inferenc script
- py_folder/sft_infer_crello.py : Refine inferenc script

Here is an example: 

```python
python zzz_infer_crello.py --out "/data1/zb/LLaMA-Factory-main/temp_out/debug" --card "6" --lora "simple_sft_17e_nocot_all_old" --base  /data1/zb/ckpts/llava13b &
python zzz_infer_crello.py --out "/data1/zb/LLaMA-Factory-main/temp_out/debug" --card "7" --lora "simple_sft_17e_nocot_all_old" --base /data1/zb/ckpts/llava13b 
```

## BibTeX

If you find this work helpful, please cite our work:
```
@inproceedings{wang2025sega,
  title={SEGA: A Stepwise Evolution Paradigm for Content-Aware Layout Generation with Design Prior},
  author={Wang, Haoran and Zhao, Bo and Wang, Jinghui and Wang, Hanzhang and Yang, Huan and Ji, Wei and Liu, Hao and Xiao, Xinyan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19321--19330},
  year={2025}
}
```

