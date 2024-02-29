import torch
from diffusers import StableDiffusionXLPipeline
from scheduling_tcd import TCDScheduler 

device = "cuda"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

prompt = "Beautiful woman, bubblegum pink, lemon yellow, minty blue, futuristic, high-detail, epic composition, watercolor."

image = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0,
    # Eta (referred to as `gamma` in the paper) is used to control the stochasticity in every step.
    # A value of 0.3 often yields good results.
    # We recommend using a higher eta when increasing the number of inference steps.
    eta=0.3, 
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]
