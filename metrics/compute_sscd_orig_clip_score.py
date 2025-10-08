from torchvision import transforms
import torch
import torch.nn as nn
import os
import sys
from PIL import Image
from tqdm import tqdm
import argparse
import pandas as pd
import re
import glob
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import open_clip


def measure_CLIP_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)


def measure_SSCD_similarity(gt_images, images, model, device):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    ret_transform = transforms.Compose([
        transforms.Resize([320, 320]),
        transforms.ToTensor(),
        normalize,
    ])

    gt_images = torch.stack([ret_transform(x.convert("RGB")) for x in gt_images]).to(
        device
    )
    images = torch.stack([ret_transform(x.convert("RGB")) for x in images]).to(device)

    with torch.no_grad():
        feat_1 = model(gt_images).clone()
        feat_1 = nn.functional.normalize(feat_1, dim=1, p=2)

        feat_2 = model(images).clone()
        feat_2 = nn.functional.normalize(feat_2, dim=1, p=2)

        return torch.mm(feat_1, feat_2.T)


def compute_sscd_and_clip_scores(args):
    # similarity model
    sim_model = torch.jit.load("download/sscd_disc_mixup.torchscript.pt").cuda().eval()

    # reference model
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cuda",
    )
    ref_model.eval() # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    ref_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    torch.set_num_threads(4)

    gt_clip_scores = []
    gen_clip_scores = []
    SSCD_sims = []

    if 'prompts' in args:
        df = pd.read_csv(args.prompts, sep=';')
    else:
        print('No prompt file provided.')

    all_gen_image_files = sorted([f for f in os.listdir(args.gen_folder) if re.search(r'.(png|jpg)', f)])
    num_iters = len(all_gen_image_files) // args.num_samples
    all_gen_images = []
    for idx in range(num_iters):
        gen_images = []
        gen_image_files = all_gen_image_files[args.num_samples*idx:args.num_samples*(idx+1)]
        for i in range(args.num_samples):
            image_real = Image.open(os.path.join(args.gen_folder, gen_image_files[i]))
            gen_images.append(image_real)
        all_gen_images.append(gen_images)

    all_gt_images = []
    for index in df["Index"]:
        gt_images = []
        for filename in glob.glob(f"sdv1_500_mem_groundtruth/gt_images/{index}/*.png"):
            im = Image.open(filename)
            gt_images.append(im)
        all_gt_images.append(gt_images)

    assert len(all_gen_images) == len(all_gt_images), 'Number of images in the generation folder and reference folder should be the same'
    
    for idx in tqdm(range(len(all_gt_images))):
        gen_images = all_gen_images[idx]
        gt_images = all_gt_images[idx]

        ### SSCD sim
        SSCD_sim = measure_SSCD_similarity(gt_images, gen_images, sim_model, "cuda")
        gt_image = gt_images[SSCD_sim.argmax(dim=0)[0].item()]
        SSCD_sim = SSCD_sim.max(dim=0).values
        SSCD_sims.append(SSCD_sim)

        ### clip score
        gt_prompt = df.iloc[idx]["Caption"]
        sims = measure_CLIP_similarity(
            [gt_image] + gen_images,
            gt_prompt,
            ref_model,
            ref_clip_preprocess,
            ref_tokenizer,
            "cuda",
        )

        gt_clip_score = sims[0:1] # gt_clip_score.shape = [1] (# of gt_image)
        gen_clip_score = sims[1:] # gen_clip_score.shape = [10] (# of gen_images)

        gt_clip_scores.append(gt_clip_score)
        gen_clip_scores.append(gen_clip_score)

    # compute SSCD
    SSCD_sims = torch.stack(SSCD_sims)
    SSCD_sims_mean = SSCD_sims.mean(dim=0)
    SSCD_sims_mean_over_seeds = SSCD_sims_mean.mean()
    print(f'Mean SSCD score: {SSCD_sims_mean_over_seeds}')

    # compute CLIP score (ground truth)
    gt_clip_scores = torch.stack(gt_clip_scores)
    gt_clip_scores_mean = gt_clip_scores.mean(dim=0)
    gt_clip_scores_mean_over_seeds = gt_clip_scores_mean.mean()
    print(f'Mean CLIP score (gt): {gt_clip_scores_mean_over_seeds}')

    # compute CLIP score (generated images)
    gen_clip_scores = torch.stack(gen_clip_scores)
    gen_clip_scores_mean = gen_clip_scores.mean(dim=0)
    gen_clip_scores_mean_over_seeds = gen_clip_scores_mean.mean()
    print(f'Mean CLIP score (gen): {gen_clip_scores_mean_over_seeds}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--gen_folder', type=str, help='Folder containing the generated images')
    parser.add_argument('--num_samples', default=10, type=int, help='Number of samples per prompt to compute the similarity score (Default: 10)')
    parser.add_argument('-p', '--prompts', type=str, help='csv file containing the prompts')

    args = parser.parse_args()
    
    compute_sscd_and_clip_scores(args)