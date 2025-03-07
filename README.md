# Seg-Zero: Reasoning-Chain Guided  Segmentation via Cognitive Reinforcement

Data: [🤗 RefCOCOg-2K](https://huggingface.co/datasets/Ricky06662/refCOCOg_2k_840) 

Model: [🤗 Seg-Zero-3B](https://huggingface.co/Ricky06662/Seg-Zero-3B)
 
Overview of Seg-Zero:

<div align=center>
<img width="98%" src="assets/overview.png"/>
</div>

Seg-Zero demonstrates following features:
1. Seg-Zero exhibits emergent test-time reasoning ability. It generates a reasoning chain before producing the final segmentation mask. 
2. Seg-Zero is trained exclusively using reinforcement learning, without any explicit supervised reasoning data.
3. Compared to supervised fine-tuning, our Seg-Zero achieves superior performance on both in-domain and out-of-domain data.




## News
[March 8th, 2024] 🔥 Seg-Zero is coming! We have released the code and training data. Paper is comming soon.


## Contents
- [Model](#model)
- [Examples](#examples)
- [Installation](#installation)
- [Inference](#inference)
- [Training](#training)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)



## Model
<div align=center>
<img width="98%" src="assets/pipeline.png"/>
</div>

Seg-Zero employs a decoupled architecture, including a reasoning model and segmentation model. We manually design a sophiscated reward mechanism that integrates both the format and accuracy rewards.


## Examples

<div align=center>
<img width="98%" src="assets/examples.png"/>
</div>


## Installation

```bash
git clone https://github.com/dvlab-research/Seg-Zero.git
cd Seg-Zero
conda create -n seg_zero python=3.11
conda activate seg_zero
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -e .
pip install sam2
pip install matplotlib
```


## Inference
```bash
python inference_scripts/infer.py
```
You can also provide your own image_path and text by:
```bash
python inference_scripts/infer.py --image_path "your_image_path" --text "your question text"
```

## Training

### 1. GRPO Training

```bash
bash training_scripts/run_qwen2_5_3b_refCOCOg.sh
```

### 2. Merge Checkpoint in Hugging Face Format

```bash
python3 training_scripts/model_merger.py --local_dir [path_to_your_actor_checkpoint]
```

> [!TIP]
> If you encounter issues with connecting to Hugging Face, consider using `export HF_ENDPOINT=https://hf-mirror.com`.


## The GRPO Algorithm

<div align=center>
<img width="48%" src="assets/rl_sample.png"/>
</div>

Seg-Zero generates several samples, calculates the rewards and then optimizes towards samples that achieve higher rewards.

> [!TIP]
> To learn more about the GRPO algorithm, you can refer to [Hugging Face's blog](https://huggingface.co/docs/trl/v0.15.2/en/grpo_trainer).


## Citation

```bibtex
@misc{liu2025segzero,
  title        = {Seg-Zero: Reasoning-Chain Guided  Segmentation via Cognitive Reinforcement},
  author       = {Liu, Yuqi and Peng, Bohao and Zhong, Zhisheng and Yue, Zihao and Yu, Bei and Jia, Jiaya},
  howpublished = {\url{https://github.com/dvlab-research/Seg-Zero}},
  year         = {2025}
}
```

## Acknowledgement
We would like to thank the following repos for their great work: 

- This work is built upon the [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl).
- This work utilizes models from  [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and [SAM2](https://huggingface.co/facebook/sam2-hiera-large).