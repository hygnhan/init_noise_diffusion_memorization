import argparse
import sys
import json
import random
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import os
from local_sd_pipeline import LocalStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel


def set_random_seed(seed=2):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_images(prompts, pipe, args):
    with torch.no_grad():
        outputs = pipe(
            prompt=prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.n_samples_per_prompt,
            method=args.method,
            args=args,
        )
        pil_images = outputs.images
        return pil_images


def setup_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.unet_id is not None:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_id, torch_dtype=torch.bfloat16
        )
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            unet=unet,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def setup_output_path(args):
    """Setup and create output directory"""
    if args.method is None:
        output_dir = 'no_mitigation'
    elif args.method == 'adj_init_noise':
        if args.per_sample:
            output_dir = f'per_sample'
        elif args.batch_wise:
            output_dir = f'batch_wise'
    output_path = os.path.join(args.output_path, output_dir)

    if args.method is None:
        hyperparams = f'g{args.guidance_scale}'
    elif args.method == 'adj_init_noise':
        if args.per_sample:
            hyperparams = f'g{args.guidance_scale}_lr{args.lr}_tl{args.target_loss}_oi{args.optim_iters}'
        elif args.batch_wise:
            hyperparams = f'g{args.guidance_scale}_rho{args.rho}_gamma{args.gamma}_M{args.adj_iters}_step{args.apply_cfg_step}'
    hyperparams = f'{hyperparams}_seed{args.seed}'

    output_path = os.path.join(output_path, hyperparams)
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, "config.json"), "w") as outfile:
        args_to_save = vars(args)
        args_to_save['command'] = " ".join(sys.argv)
        json.dump(args_to_save, outfile)
    return output_path


def load_prompt_csv_file(args):
    """Load prompt csv file based on arguments"""
    df = pd.read_csv('prompts/memorized_laion_prompts.csv', sep=';')
    print("Loaded CSV file")
    return df


def main(args):
    # Load diffusion model
    pipe = setup_model(args)
    
    # Setup random seed
    set_random_seed(args.seed)

    # Setup output path
    output_path = setup_output_path(args)
    print(f"Save images in dir: {output_path}")

    # Load prompt csv file
    df = load_prompt_csv_file(args)
    
    # Generate images
    for i in tqdm(range(len(df) // args.batch_size), total=len(df) // args.batch_size):
        rows = df.iloc[i*args.batch_size:(i+1)*args.batch_size]
        prompts = rows['Caption'].to_list()

        images = generate_images(prompts, pipe, args)
        for j in range(len(images)):
            images[j].save(f"{output_path}/img_{i*args.batch_size + j // args.n_samples_per_prompt:04d}_{j%args.n_samples_per_prompt:02d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating images")
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--unet_id", default=None)

    parser.add_argument('--output_path', default='generated_images', type=str, 
                        help="output folder for generated images (default: \'generated_images\')")
    
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='Number of prompts per batch')
    parser.add_argument('--n_samples_per_prompt', default=10, type=int, help='number of generated samples for each prompt (default: 10)')
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--guidance_scale", default=7, type=float)
    parser.add_argument("--seed", default=2, type=int)

    # Mitigation strategies
    parser.add_argument("--method", default=None, choices=[None, 'adj_init_noise'], type=str)
    
    # Ours (per-sample)
    parser.add_argument("--per_sample", action="store_true")
    parser.add_argument("--target_loss", default=0.9, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--optim_iters", default=1000, type=int)

    # Ours (batch-wise)
    parser.add_argument("--batch_wise", action="store_true")
    parser.add_argument("--rho", default=50, type=float)
    parser.add_argument("--gamma", default=0.7, type=float)
    parser.add_argument("--adj_iters", default=2, type=int)
    parser.add_argument("--apply_cfg_step", default=4, type=int)

    args = parser.parse_args()

    main(args)
