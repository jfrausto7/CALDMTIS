from config import API_KEY, IMAGE_DIR, ACCESS_TOKEN
from metrics import calculate_clip_score, calculate_niqe, calculate_brisque, calculate_teng, calculate_gmsd, aggregate_scores
from utils import generate_violinplot, generate_stripplot, generate_GMSD_heatmap, generate_correlation_heatmap
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch
import os
import numpy as np
import google.generativeai as palm
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Initialize partial function for calculating CLIP score
# The model used is "openai/clip-vit-base-patch16"
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

# Models names
model_names = ["base1_0", "refined1_0", "base0_9", "refined0_9", "two_one", "one_five"]

# Initialize lists for storing scores and prompts
if len(os.listdir("numpy")) != 0:
    CLIP_scores, NIQE_scores, BRISQUE_scores, TENG_scores = [], [], [], []
    for model in model_names:
        CLIP_scores.append(np.load(f"numpy/{model}CLIPscores.npy"))
        NIQE_scores.append(np.load(f"numpy/{model}NIQEscores.npy"))
        BRISQUE_scores.append(np.load(f"numpy/{model}BRISQUEscores.npy"))
        TENG_scores.append(np.load(f"numpy/{model}TENGscores.npy"))
    GMSD_matrices = np.load(f"numpy/GMSDmatrices.npy")
    prompts = np.load("numpy/prompts.npy")

else:
    CLIP_scores, NIQE_scores, BRISQUE_scores, TENG_scores = [], [], [], []
    for model in model_names:
        CLIP_scores.append(np.array([]))
        NIQE_scores.append(np.array([]))
        BRISQUE_scores.append(np.array([]))
        TENG_scores.append(np.array([]))
    GMSD_matrices = np.array([])
    prompts = np.array([])

# Load diffusion models: base models, refiner models, and old models
models = []
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
models.append(pipe)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
models.append(refiner)
pipe0_9 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_auth_token=ACCESS_TOKEN)
models.append(pipe0_9)
refiner0_9 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, variant="fp16", use_auth_token=ACCESS_TOKEN) 
models.append(refiner0_9)
pipe2_1 = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
models.append(pipe2_1)
pipe2_1.scheduler = DPMSolverMultistepScheduler.from_config(pipe2_1.scheduler.config)
pipe1_5 = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
models.append(pipe1_5)

# Enable model CPU offload for all models
for model in models:
    model.enable_model_cpu_offload()

for i in range(5):
    # Configure PaLM; generate prompts & image names
    palm.configure(api_key=API_KEY)
    prompt = palm.generate_text(prompt="In less than 77 words, craft a thought-provoking and imaginative prompt that challenges a text-to-image diffusion model to create captivating and evocative visuals. Explore an array of themes, emotions, and scenes to inspire a diverse range of images that resonate with creativity and intrigue. Embrace a multitude of subjects, contexts, and concepts to infuse the images with a rich tapestry of diversity. Do not use any stylistic formatting like asterisks in this prompt.")
    print("PROMPT: " + prompt.result)
    imageName = palm.generate_text(prompt="Given the following text-to-image prompt, come up with a short one-word name for its associated image file: " + prompt.result)
    path = os.path.join(IMAGE_DIR, imageName.result)
    while os.path.exists(path):
        imageName = palm.generate_text(prompt=f"Given the following text-to-image prompt, come up with a short one-word name for its associated image file. Do not use {imageName}. " + prompt.result)
        path = os.path.join(IMAGE_DIR, imageName.result)
    print("FILE NAME: " + imageName.result)
    prompt = prompt.result
    imageName = imageName.result
    prompts = np.append(prompts, prompt)

    # Generate images for the given prompt
    images = []
    for i in range(len(models)):
        if i == 1 or i == 3:
            image = models[i](prompt=prompt, image=images[-1]).images[0]
            images.append(image)
        else:
            image = models[i](prompt=prompt).images[0]
            images.append(image)

    # Save the generated images
    os.mkdir(path)
    for i in range(len(model_names)):
        images[i].save(path + '/' + model_names[i] + ".jpg")

    # Calculate and store scores for each image
    for i in range(len(images)):
        CLIP_scores[i] = np.append(CLIP_scores[i], calculate_clip_score(np.array(images[i]), prompt, clip_score_fn))
        NIQE_scores[i] = np.append(NIQE_scores[i], calculate_niqe(images[i]))
        BRISQUE_scores[i] = np.append(BRISQUE_scores[i], calculate_brisque(images[i]))
        TENG_scores[i] = np.append(TENG_scores[i], calculate_teng(images[i]))
    GMSD_matrices = np.concatenate((GMSD_matrices, [calculate_gmsd(images)]), axis=0) if GMSD_matrices.size else np.array([calculate_gmsd(images)])

    # Save scores to externally saved lists
    for i in range(len(model_names)):
        np.save(f'numpy/{model_names[i]}CLIPscores', CLIP_scores[i])
        np.save(f'numpy/{model_names[i]}NIQEscores', NIQE_scores[i])
        np.save(f'numpy/{model_names[i]}BRISQUEscores', BRISQUE_scores[i])
        np.save(f'numpy/{model_names[i]}TENGscores', TENG_scores[i])
    np.save(f"numpy/GMSDmatrices.npy", GMSD_matrices)
    np.save("numpy/prompts", prompts)
  
# Print scores and prompts
print(CLIP_scores)
print(NIQE_scores)
print(BRISQUE_scores)
print(TENG_scores)
print(GMSD_matrices)
print(f"Prompts: {prompts}")

# Generate plots
metric_scores = np.array([CLIP_scores, NIQE_scores, BRISQUE_scores, TENG_scores])
generate_violinplot(CLIP_scores, "CLIP")
generate_stripplot(CLIP_scores, "CLIP")
generate_violinplot(NIQE_scores, "NIQE")
generate_stripplot(NIQE_scores, "NIQE")
generate_violinplot(BRISQUE_scores, "BRISQUE")
generate_stripplot(BRISQUE_scores, "BRISQUE")
generate_violinplot(TENG_scores, "TENG")
generate_stripplot(TENG_scores, "TENG")
generate_GMSD_heatmap(GMSD_matrices)
generate_correlation_heatmap(metric_scores, model_names)
aggregate_scores((CLIP_scores, NIQE_scores, BRISQUE_scores, TENG_scores), model_names)