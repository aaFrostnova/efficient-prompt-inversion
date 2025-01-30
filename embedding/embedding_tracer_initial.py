# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler 
from transformers import CLIPTextModel, CLIPTokenizer, AutoProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
#from torch.cuda.amp import GradScaler
from inference_utils import SSIMLoss,psnr,lpips_fn,save_img_tensor
from inference_models_multilatent import get_init_noise, get_model,from_noise_to_image
from inference_image0_multilatent import get_image0
import json
from sklearn.metrics.pairwise import cosine_similarity
#import ot
#from torch_two_sample import SmoothKNNStatistic, SmoothFRStatistic
import numpy as np
from typing import Optional, Union, Tuple, List, Callable, Dict
import sys
import argparse
from numba import jit
from collections import Counter
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import pickle
import PIL.Image as Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
matplotlib.use('agg')

parser = argparse.ArgumentParser()
parser.add_argument("--input_selection", default="", type=str, help="The path of dev set.")
parser.add_argument("--model_type", default="ddpm_cifar10", type=str, help="The path of dev set.")
parser.add_argument("--model_path_", default=None, type=str, help="The path of dev set.")

parser.add_argument("--dataset_index", default=None, type=int, help="")
parser.add_argument("--bs", default=8, type=int, help="")
parser.add_argument("--write_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--num_iter", default=1000, type=int, help="The path of dev set.")
parser.add_argument("--input_selection_url", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_name", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_model_type", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_model_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--re_loss_list_savename", default=None, type=str, help="The path of dev set.")
parser.add_argument("--init_gt_dist", default=None, type=float, help="The path of dev set.")
parser.add_argument("--generation_size", default=None, type=int, help="The path of dev set.")
parser.add_argument("--name", default=None, type=str, help="current image name")
args = parser.parse_args()
from torch.nn.parallel import DataParallel
device = 'cuda:0'
model_path = "SfinOe/stable-diffusion-v1.5"
blip_path = "Salesforce/blip-image-captioning-large"
text_encoder = CLIPTextModel.from_pretrained(model_path+"/text_encoder")
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
tokenizer = CLIPTokenizer.from_pretrained(model_path+"/tokenizer")
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device)
blip_processor = AutoProcessor.from_pretrained(blip_path)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_path, torch_dtype=torch.float16).to(device)
processors = [blip_processor]
caption_models = [blip_model]

args.cur_model = get_model(args.model_type,args.model_path_,args)
image0, gt_noise = get_image0(args, vae)
image0 = image0.detach()
args.image0 = image0


class StableDiffusionModel(nn.Module):
    def __init__(self, text_encoder, unet, tokenizer):
        super(StableDiffusionModel, self).__init__()
        self.text_encoder = text_encoder
        self.unet = unet
        self.tokenizer = tokenizer

    def forward(self, text_embedding):
        text_embedding = text_embedding.to(device)
        height = 512  # default height of Stable Diffusion
        width = 512   # default width of Stable Diffusion
        num_inference_steps = 5 # Number of denoising steps
        guidance_scale = 7.5
        
        # init the noise 
        generator = torch.manual_seed(0)
        init_noise = torch.randn((1, self.unet.in_channels, height // 8, width // 8), generator=generator).to(device)
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps)
        init_noise = init_noise * scheduler.init_noise_sigma

        # init the text embedding
        uncond_input = self.tokenizer([""], padding="max_length", max_length=len(text_embedding[0]), return_tensors="pt")
        uncond_embedding = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embedding, text_embedding])

        

        for t in tqdm(scheduler.timesteps):

            latent_model_input = torch.cat([init_noise] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # check noise_pred
            init_noise = scheduler.step(noise_pred, t, init_noise).prev_sample
            latents_new = init_noise
        
        return latents_new
# init model

model = StableDiffusionModel(text_encoder, unet, tokenizer).to(device[0])
criterion = torch.nn.MSELoss(reduction='none') 



def optimize_embedding(init_embedding, num_iter, end=16):
    cur_init_embedding = torch.nn.Parameter(init_embedding[:,1:end,:]).cuda(0)
    optimizer = torch.optim.AdamW([cur_init_embedding], lr=0.1)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.05, max_lr=0.1, step_size_up=50, step_size_down=None, cycle_momentum=False, mode='triangular')
    best_loss = 9999
    embedding_history = []
    loss_history = []
    best_embedding = init_embedding
    best_embeddings = []
    for i in range(num_iter):
        embedding_history.append(cur_init_embedding.detach().cpu().numpy())
        generated_latent = model(torch.cat([init_embedding[:,:1,:], cur_init_embedding], dim=1))
        image = from_noise_to_image(args,args.cur_model,generated_latent,args.model_type)
        loss = criterion(image0.detach(), image).mean()
        loss_history.append(loss.item())
        print(f"Iteration {i}, Loss: {loss.item()}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_embedding = cur_init_embedding.detach().cpu().numpy()
            best_loss = loss.item()
        if (i+1) % 50 == 0 or i == 0:
            best_embeddings.append(best_embedding.tolist())

    
    return np.array(embedding_history), best_embeddings

num_iter = args.num_iter


from torchvision import transforms
image0_pil = transforms.ToPILImage()(image0.squeeze(0))
generated_texts = []

for processor, caption_model in zip(processors, caption_models):
    device_cur = next(caption_model.parameters()).device
    inputs = processor(image0_pil, return_tensors="pt").to(device_cur, torch.float16)
    generated_ids = caption_model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)
    generated_texts.append(generated_text)
losses = []
for generated_text in generated_texts:
    text_input = tokenizer(generated_text, padding="max_length", max_length=32, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device[0]))[0]
    generated_latent = model(text_embeddings)
    image = from_noise_to_image(args,args.cur_model,generated_latent,args.model_type)
    loss = criterion(image0.detach(), image).mean()
    losses.append(loss)

init_prompt = generated_texts[losses.index(min(losses))]

text_input = tokenizer(init_prompt, padding="max_length", max_length=32, truncation=True, return_tensors="pt")
attention_mask = text_input.attention_mask[0]
end = torch.where(attention_mask == 0)[0][0] if 0 in attention_mask else len(attention_mask)
print(end)
with torch.no_grad():
    embedding = text_encoder(text_input.input_ids.to(device[0]))[0]

# Store all optimization paths
all_embedding_histories = []
best_embeddings_list = []
# Optimize for each initial point
print(f"Optimizing embedding for prompt: {init_prompt}")
embedding_history, best_embeddings = optimize_embedding(embedding, num_iter, end)
best_embeddings_list.append(best_embeddings)
all_embedding_histories.append(embedding_history)

for i, best_embeddings in enumerate(best_embeddings_list):
    print(args.name)
    with open(f'{args.write_path}/embedding_rec_{args.name}_{i}.json', 'w') as file:
        json.dump(best_embeddings, file)