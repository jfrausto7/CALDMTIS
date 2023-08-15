from config import API_KEY, IMAGE_DIR, ACCESS_TOKEN
from metrics import calculate_clip_score, calculate_niqe, calculate_brisque, calculate_teng, calculate_gmsd
from utils import generate_violinplot, generate_stripplot
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch
import os
import numpy as np
import google.generativeai as palm
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Initialize partial function for calculating CLIP score
# https://arxiv.org/abs/2104.08718
# The model used is "openai/clip-vit-base-patch16"
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

# Initialize lists for storing scores and prompts
if os.path.isfile("baseCLIPscores.npy"):
    baseCLIPscores = np.load("numpy/baseCLIPscores.npy") 
    refinedCLIPscores = np.load("numpy/refinedCLIPscores.npy") 
    base0_9CLIPscores = np.load("numpy/base0_9CLIPscores.npy") 
    refined0_9CLIPscores = np.load("numpy/refined0_9CLIPscores.npy") 
    one_fiveCLIPscores = np.load("numpy/one_fiveCLIPscores.npy")
    two_oneCLIPscores = np.load("numpy/two_oneCLIPscores.npy")
    prompts = np.load("numpy/prompts.npy")
else:
    baseCLIPscores = np.array([])
    refinedCLIPscores = np.array([])
    base0_9CLIPscores = np.array([])
    refined0_9CLIPscores = np.array([])
    one_fiveCLIPscores = np.array([])
    two_oneCLIPscores = np.array([])
    prompts = np.array([])

for i in range(1):
    # Configure PaLM; generate prompts & image names
    palm.configure(api_key=API_KEY)
    prompt = palm.generate_text(prompt="In less than 77 words, come up with an incredibly in-depth, random, and unique prompt for a text-to-image model. Make sure you haven't generated it before; it needs to be completely original.")
    print("PROMPT: " + prompt.result)
    imageName = palm.generate_text(prompt="Given the following text-to-image prompt, come up with a short one-word name for its associated image file: " + prompt.result)
    print("FILE NAME: " + imageName.result)
    prompt = prompt.result
    imageName = imageName.result
    prompts = np.append(prompts, prompt)

    # Load diffusion models: base model, refiner model, and old model
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe0_9 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_auth_token=ACCESS_TOKEN)
    refiner0_9 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, variant="fp16", use_auth_token=ACCESS_TOKEN) 
    pipe1_5 = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe2_1 = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    pipe2_1.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Enable model CPU offload for all models
    pipe.enable_model_cpu_offload()
    refiner.enable_model_cpu_offload()
    pipe0_9.enable_model_cpu_offload()
    refiner0_9.enable_model_cpu_offload()
    pipe1_5.enable_model_cpu_offload()
    pipe2_1.enable_model_cpu_offload()

    # Generate images for the given prompt
    image = pipe(prompt=prompt).images[0]
    imageRefined = refiner(prompt=prompt, image=image).images[0]
    image0_9 = pipe0_9(prompt=prompt).images[0]
    imageRefined0_9 = refiner0_9(prompt=prompt, image=image0_9).images[0]
    image1_5 = pipe1_5(prompt=prompt).images[0]
    image2_1 = pipe2_1(prompt=prompt).images[0]

    # Save the generated images
    path = os.path.join(IMAGE_DIR, imageName)
    os.mkdir(path)
    image.save(path + "/base.jpg")
    imageRefined.save(path + "/refined.jpg")
    image0_9.save(path + "/base0_9.jpg")
    imageRefined0_9.save(path + "/refined0_9.jpg")
    image1_5.save(path + "/1_5.jpg")
    image2_1.save(path + "/2_1.jpg")

    # Calculate and store CLIP scores for each image
    # TODO: Clean up this mess!
    sd_clip_score = calculate_clip_score(np.array(image), prompt, clip_score_fn)
    sd_clip_score_REFINED = calculate_clip_score(np.array(imageRefined), prompt, clip_score_fn)
    sd_clip_score_0_9 = calculate_clip_score(np.array(image0_9), prompt, clip_score_fn)
    sd_clip_score_REFINED_0_9 = calculate_clip_score(np.array(imageRefined0_9), prompt, clip_score_fn)
    sd_clip_score_1_5 = calculate_clip_score(np.array(image1_5), prompt, clip_score_fn)
    sd_clip_score_2_1 = calculate_clip_score(np.array(image2_1), prompt, clip_score_fn)
    baseCLIPscores = np.append(baseCLIPscores, sd_clip_score)
    refinedCLIPscores = np.append(refinedCLIPscores, sd_clip_score_REFINED)
    base0_9CLIPscores = np.append(base0_9CLIPscores, sd_clip_score_0_9)
    refined0_9CLIPscores = np.append(refined0_9CLIPscores, sd_clip_score_REFINED_0_9)
    one_fiveCLIPscores = np.append(one_fiveCLIPscores, sd_clip_score_1_5)
    two_oneCLIPscores = np.append(two_oneCLIPscores, sd_clip_score_2_1)
    # TODO: Configure metrics for all models
    print("NIQE: " + str(calculate_niqe(image)))
    print("BRISQUE: " + str(calculate_brisque(image)))
    print("TENG: " + str(calculate_teng(image)))
    gmsd_matrix = calculate_gmsd([image, imageRefined, image0_9, imageRefined0_9, image1_5, image2_1])
    print("GMSD: ")
    print(gmsd_matrix)

    # Save scores to externally saved lists
    np.save('numpy/baseCLIPscores', baseCLIPscores)
    np.save('numpy/refinedCLIPscores', refinedCLIPscores)
    np.save('numpy/base0_9CLIPscores', base0_9CLIPscores)
    np.save('numpy/refined0_9CLIPscores', refined0_9CLIPscores)
    np.save('numpy/one_fiveCLIPscores', one_fiveCLIPscores)
    np.save('numpy/two_oneCLIPscores', two_oneCLIPscores)
    np.save("numpy/prompts", prompts)
  
# Print CLIP scores and prompts
print(f"CLIP scores base: {baseCLIPscores}")
print(f"CLIP scores refined: {refinedCLIPscores}")
print(f"CLIP scores base 0.9: {base0_9CLIPscores}")
print(f"CLIP scores refined 0.9: {refined0_9CLIPscores}")
print(f"CLIP scores 1.5: {one_fiveCLIPscores}")
print(f"CLIP scores 2.1: {two_oneCLIPscores}")
print(f"Prompts: {prompts}")
generate_violinplot(baseCLIPscores, refinedCLIPscores, base0_9CLIPscores, refined0_9CLIPscores, one_fiveCLIPscores, two_oneCLIPscores)
generate_stripplot(baseCLIPscores, refinedCLIPscores, base0_9CLIPscores, refined0_9CLIPscores, one_fiveCLIPscores, two_oneCLIPscores)