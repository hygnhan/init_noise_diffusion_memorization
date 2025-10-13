##  Adjusting Initial Noise to Mitigate Memorization in Text-to-Image Diffusion Models
This repository provides the official PyTorch implementation of the following paper:
> Adjusting Initial Noise to Mitigate Memorization in Text-to-Image Diffusion Models<br>
> Hyeonggeun Han* (Seoul National University), Sehwan Kim* (Seoul National University), Hyungjun Joo (Seoul National University), Sangwoo Hong (Konkuk University), Jungwoo Lee (Seoul National University, Hodoo AI Labs)<br>
> \* These authors contributed equally to this work.<br>
> NeurIPS 2025

> Paper: [Arxiv](https://arxiv.org/pdf/2510.08625)<br>

**Abstract:** 
*Despite their impressive generative capabilities, text-to-image diffusion models often memorize and replicate training data, prompting serious concerns over privacy and copyright. Recent work has attributed this memorization to an attraction basin—a region where applying classifier-free guidance (CFG) steers the denoising trajectory toward memorized outputs—and has proposed deferring CFG application until the denoising trajectory escapes this basin. However, such delays often result in non-memorized images that are poorly aligned with the input prompts, highlighting the need to promote earlier escape so that CFG can be applied sooner in the denoising process. In this work, we show that the initial noise sample plays a crucial role in determining when this escape occurs. We empirically observe that different initial samples lead to varying escape times. Building on this insight, we propose two mitigation strategies that adjust the initial noise—either collectively or individually—to find and utilize initial samples that encourage earlier basin escape. These approaches significantly reduce memorization while preserving image-text alignment.*<br>

## Dependencies
- Python == 3.8.0 
- PyTorch == 2.3.1
- transformers == 4.41.2
- diffusers == 0.18.2
- clip
- opencv-python
- datasets

## Pytorch Implementation
We provide the implementation code of our proposed methods. To evaluate them, first generate images using the provided generation code. After the images are generated, you can run the evaluation code to assess their performance. Detailed instructions for both image generation and evaluation are provided below.

### Image generation

#### Run No Mitigation
```
python generate_images.py --seed 2
```

#### Run Batch-wise mitigation (ours)
```
python generate_images.py --seed 2 --method adj_init_noise --batch_wise --rho 50 --gamma 0.7 --adj_iters 2 --apply_cfg_step 4
```

#### Run Per-sample mitigation (ours)
```
python generate_images.py --seed 2 --method adj_init_noise --per_sample --target_loss 0.9 --lr 0.01
```

### Evaluation

#### Calculate SSCD and CLIP scores
1. Create a directory named ``download`` in the current working directory.
2. Download the checkpoint file ``sscd_disc_mixup.torchscript.pt`` from [sscd-copy-detection](https://github.com/facebookresearch/sscd-copy-detection) and place it in the ``download`` directory. Additionally, download all ground-truth images from [Detecting](https://github.com/YuxinWenRick/diffusion_memorization).

3. Run the following command:
```
python metrics/compute_sscd_orig_clip_score.py --gen_folder GEN_IMG_DIR --prompts prompts/memorized_laion_prompts.csv
```

#### Calculate LPIPS and ImageReward scores
1. Install the required packages:
```
pip install lpips
pip install image-reward
```

2. Run the following command:
```
python metrics/compute_lpips_imagereward.py --gen_folder GEN_IMG_DIR --prompts prompts/memorized_laion_prompts.csv
```

## Contact
Hyeonggeun Han (hygnhan@snu.ac.kr)

Sehwan Kim (sehwankim@snu.ac.kr)

## Acknowledgments
Our PyTorch implementation is built upon [Detecting](https://github.com/YuxinWenRick/diffusion_memorization) and [NeMo](https://github.com/ml-research/localizing_memorization_in_diffusion_models). We sincerely thank the authors for their foundational work.
