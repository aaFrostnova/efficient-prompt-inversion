import sys
sys.path.append('../')
import vec2text
import json
import torch
import os
import open_clip
from tqdm.auto import tqdm
from lpips import LPIPS
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, CLIPTokenizer, CLIPTextModel
import bert_score
from diffusers import PNDMScheduler, StableDiffusionPipeline
from torch import nn
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor()
])

##### set your path here
inversion_model_path = "path to your zero model"
corrector_model_path = "path to your correction model"
model_id = "stable-diffusion-v1-5"
embedding_path = "path to your reconstructed embeddings folder"
origin_image_path = "path to your target images"

device = "cuda" if torch.cuda.is_available() else "cpu"
inversion_model = vec2text.models.InversionModel.from_pretrained(inversion_model_path)
corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(corrector_model_path)
corrector = vec2text.load_corrector(inversion_model, corrector_model)
tokenizer = CLIPTokenizer.from_pretrained(model_id+"/tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id+"/text_encoder")

lpips_model = LPIPS(net='alex')
 # Setup CLIP model
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
# Setup Stable Diffusion pipeline
scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16).to(device)

ppl_model_id = "openai-community/gpt2"
ppl_model = GPT2LMHeadModel.from_pretrained(ppl_model_id).to(device)
ppl_tokenizer = GPT2TokenizerFast.from_pretrained(ppl_model_id)

def ppl(text):
    max_length = ppl_model.config.n_positions
    stride = 10
    encodings = ppl_tokenizer("\n\n".join([text]), return_tensors="pt")

    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = ppl_model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl

def get_best_text(
    image_path,
    prompts,
    pipe=pipe,
    output_dir='./logs',
    device="cuda",
    image_length=512,
    num_prompts=20,
    seed=0
    ):
   

    # Setup generator
    generator = torch.Generator(device=device).manual_seed(seed)

    # Load original image
    orig_image = Image.open(image_path).convert('RGB')

    best_loss_clip = 0.
    best_loss_lpips = 1.
    best_text = ""
    best_pred = None


    # Process prompts
    for step, prompt in enumerate(tqdm(prompts[:num_prompts]), 1):
        with torch.no_grad():
            # Generate image from prompt
            pred_imgs = pipe(
                prompt,
                num_images_per_prompt=1,
                guidance_scale=7.5,
                num_inference_steps=50,
                height=image_length,
                width=image_length,
                generator=generator
            ).images

            # Prepare images for CLIP similarity measurement
            orig_images_t = torch.cat([clip_preprocess(orig_image).unsqueeze(0)]).to(device)
            pred_imgs_t = torch.cat([clip_preprocess(i).unsqueeze(0) for i in pred_imgs]).to(device)

            # Measure CLIP similarity
            eval_loss_clip = measure_clip_similarity(orig_images_t, pred_imgs_t, clip_model, device)

            orig_tensor = transform(orig_image).unsqueeze(0) 
            pred_tensor = transform(pred_imgs[0]).unsqueeze(0)
            eval_loss_lpips = lpips_model(orig_tensor, pred_tensor)


            # Update best result if necessary
            if best_loss_clip < eval_loss_clip:
                best_loss_clip = eval_loss_clip
                best_loss_lpips = eval_loss_lpips
                best_text = prompt
                best_pred = pred_imgs[0]

    # Save results
    # best_pred.save(f'{output_dir}/pred_image.png')
    print(f"\nBest shot: cosine similarity: {best_loss_clip:.3f}")
    print(f"\nBest shot: lpips similarity: {best_loss_lpips.item():.3f}")
    print(f"text: {best_text}")

    return best_loss_clip, best_loss_lpips.item(), best_text

def measure_clip_similarity(orig_images, pred_images, clip_model, device):
    with torch.no_grad():
        orig_feat = clip_model.encode_image(orig_images)
        orig_feat = orig_feat / orig_feat.norm(dim=1, keepdim=True)

        pred_feat = clip_model.encode_image(pred_images)
        pred_feat = pred_feat / pred_feat.norm(dim=1, keepdim=True)
        return (orig_feat @ pred_feat.t()).mean().item()



json_files = [f for f in os.listdir(embedding_path) if f.endswith('image.json')]
best_texts = {}
with open(os.path.join(origin_image_path, "prompts.json"), 'r') as file:
    meta_data = json.load(file)
for json_file in json_files: 
    image_name = json_file.split("_")[2]
    image_path =  image_name + ".png" 
    file_path = os.path.join(embedding_path, json_file)
    with open(file_path, 'r') as f:
        init_embeddings = json.load(f)
    results = []
    
    for init_embedding in init_embeddings:
        init_embedding = torch.tensor(init_embedding)
        last_hidden_state = init_embedding.to("cuda")
        pool_position = torch.tensor([-1]).to("cuda")
        pooled_output = last_hidden_state[
                            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                            pool_position,
                        ]
        
        result = vec2text.invert_embeddings(
                embeddings=pooled_output,
                corrector=corrector,
                num_steps=20,
                sequence_beam_width=4,
            )
        with open(os.path.join(embedding_path, f"{image_name}.txt"), "a") as txt_file:
            txt_file.write(result[0]+'\n')
        results.append(result)
    

    best_loss_clip, best_loss_lpips, best_text = get_best_text(os.path.join(origin_image_path,image_path), results, pipe)
    P, R, F1 = bert_score.score(best_text, [meta_data[image_name + ".png"]], lang="en", verbose=True,model_type='bert-large-uncased')
    ppl_score = ppl(best_text[0])
    # Create or load existing results
    result_file = os.path.join(embedding_path, "result.json")    
    # Add new result
    best_texts[image_path] = {
        "text": best_text,
        "cos_sim": best_loss_clip, 
        "lpips_sim": best_loss_lpips,
        "P": P.item(),
        "R": R.item(), 
        "F1": F1.item(),
        "ppl": ppl_score.item()
    }
    
    # Save after each update
    with open(result_file, 'w') as file:
        json.dump(best_texts, file, indent=4)
