import numpy as np
import torch
import os
import sys
from PIL import Image
from tqdm import tqdm
import argparse
import pandas as pd
import re
import glob
import lpips
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import ImageReward as RM

device = "cuda"
reward_model = RM.load("ImageReward-v1.0")


def compute_image_similarity_scores(args):
    torch.set_num_threads(4)

    gen_imagerw_scores = []
    gen_lpips_scores = []

    if 'prompts' in args:
        df = pd.read_csv(args.prompts, sep=';')
    else:
        print('No prompt file provided.')


    all_gen_image_files = sorted([f for f in os.listdir(args.gen_folder) if re.search(r'.(png|jpg)', f)])
    num_iters = len(all_gen_image_files) // args.num_samples
    all_image_path = []
    for idx in range(num_iters):
        image_path = []
        gen_image_files = all_gen_image_files[args.num_samples*idx:args.num_samples*(idx+1)]
        for i in range(args.num_samples):
            image_path.append(os.path.join(args.gen_folder, gen_image_files[i]))
        all_image_path.append(image_path)

    ### LPIPS ###
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()
    
    for i in range(len(all_image_path)):
        files = all_image_path[i]
        dists = []

        for (ff,file) in enumerate(files[:-1]):
            img0 = lpips.im2tensor(lpips.load_image(file))
            img0 = img0.cuda()
            files1 = files[ff+1:]
            
            for file1 in files1:
                img1 = lpips.im2tensor(lpips.load_image(file1))
                img1 = img1.cuda()

                # Compute distance
                dist01 = loss_fn.forward(img0,img1)
                dists.append(dist01.item())
        avg_dist = np.mean(np.array(dists))
        gen_lpips_scores.append(avg_dist)

    stacked = np.stack(gen_lpips_scores)
    print("LPIPS mean : ", np.mean(stacked, axis=0))


    all_gt_images = []
    for index in df["Index"]:
        gt_images = []
        for filename in glob.glob(f"sdv1_500_mem_groundtruth/gt_images/{index}/*.png"):
            im = Image.open(filename)
            gt_images.append(im)

        all_gt_images.append(gt_images)

    for idx in tqdm(range(len(all_gt_images))): 
        gt_prompt = df.iloc[idx]["Caption"]
      
        ### Image-reward ###
        gen_imagerw_score = reward_model.score(gt_prompt, all_image_path[idx])
        gen_imagerw_score = torch.Tensor(gen_imagerw_score)
        gen_imagerw_scores.append(gen_imagerw_score)
    
    # Compute statistics over the whole set
    gen_imagerw_scores = torch.stack(gen_imagerw_scores)
    gen_imagerw_scores_mean = gen_imagerw_scores.mean(dim=0)
    gen_imagerw_scores_mean_over_seeds = gen_imagerw_scores_mean.mean()
    print(f'Mean Image-reward (gen, Overall): {gen_imagerw_scores_mean_over_seeds}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--gen_folder', type=str, help='Folder containing the generated images')
    parser.add_argument('--num_samples', default=10, type=int, help='Number of samples per prompt to compute the similarity score (Default: 10)')
    parser.add_argument('-p', '--prompts', type=str, help='csv file containing the prompts')
    args = parser.parse_args()
    
    compute_image_similarity_scores(args)