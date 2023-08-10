from config import API_KEY, IMAGE_DIR
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
    baseCLIPscores = np.load("baseCLIPscores.npy") 
    refinedCLIPscores = np.load("refinedCLIPscores.npy") 
    one_fiveCLIPscores = np.load("one_fiveCLIPscores.npy")
    two_oneCLIPscores = np.load("two_oneCLIPscores.npy")
    prompts = np.load("prompts.npy")
else:
    baseCLIPscores = np.array([])
    refinedCLIPscores = np.array([])
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
    # TODO: ADD SDXL 0.9b+r
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe1_5 = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe2_1 = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    pipe2_1.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Enable model CPU offload for all models
    pipe.enable_model_cpu_offload()
    refiner.enable_model_cpu_offload()
    pipe1_5.enable_model_cpu_offload()
    pipe2_1.enable_model_cpu_offload()

    # Generate images for the given prompt
    image = pipe(prompt=prompt).images[0]
    imageRefined = refiner(prompt=prompt, image=image).images[0]
    image1_5 = pipe1_5(prompt=prompt).images[0]
    image2_1 = pipe2_1(prompt=prompt).images[0]

    # Save the generated images
    path = os.path.join(IMAGE_DIR, imageName)
    os.mkdir(path)
    image.save(path + "/base.jpg")
    imageRefined.save(path + "/refined.jpg")
    image1_5.save(path + "/1_5.jpg")
    image2_1.save(path + "/2_1.jpg")

    # Calculate and store CLIP scores for each image
    sd_clip_score = calculate_clip_score(np.array(image), prompt, clip_score_fn)
    sd_clip_score_REFINED = calculate_clip_score(np.array(imageRefined), prompt, clip_score_fn)
    sd_clip_score_1_5 = calculate_clip_score(np.array(image1_5), prompt, clip_score_fn)
    sd_clip_score_2_1 = calculate_clip_score(np.array(image2_1), prompt, clip_score_fn)
    baseCLIPscores = np.append(baseCLIPscores, sd_clip_score)
    refinedCLIPscores = np.append(refinedCLIPscores, sd_clip_score_REFINED)
    one_fiveCLIPscores = np.append(one_fiveCLIPscores, sd_clip_score_1_5)
    two_oneCLIPscores = np.append(two_oneCLIPscores, sd_clip_score_2_1)
    print("NIQE: " + str(calculate_niqe(image)))
    print("BRISQUE: " + str(calculate_brisque(image)))
    print("TENG: " + str(calculate_teng(image)))
    print("GMSD: " + str(calculate_gmsd(image, imageRefined)))

    # Save scores to externally saved lists
    np.save('baseCLIPscores', baseCLIPscores)
    np.save('refinedCLIPscores', refinedCLIPscores)
    np.save('one_fiveCLIPscores', one_fiveCLIPscores)
    np.save('two_oneCLIPscores', two_oneCLIPscores)
    np.save("prompts", prompts)
  
# Print CLIP scores and prompts
print(f"CLIP scores base: {baseCLIPscores}")
print(f"CLIP scores refined: {refinedCLIPscores}")
print(f"CLIP scores 1_5: {one_fiveCLIPscores}")
print(f"CLIP scores 2_1: {two_oneCLIPscores}")
print(f"Prompts: {prompts}")
generate_violinplot(baseCLIPscores, refinedCLIPscores, one_fiveCLIPscores, two_oneCLIPscores)
generate_stripplot(baseCLIPscores, refinedCLIPscores, one_fiveCLIPscores, two_oneCLIPscores)