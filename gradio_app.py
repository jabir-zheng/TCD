import random

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline

from scheduling_tcd import TCDScheduler

device = "cuda"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()


def inference(prompt, num_inference_steps=4, seed=-1, eta=0.3):
    if seed is None or seed == '' or seed == -1:
        seed = int(random.randrange(4294967294))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=0,
        eta=eta,
        generator=generator,
    ).images[0]
    return image


# Define style
title = "<h1 style='text-align: center'>Trajectory Consistency Distillation</h1>"
description = "Official 🤗 Gradio demo for Trajectory Consistency Distillation"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/' target='_blank'>Trajectory Consistency Distillation</a> | <a href='https://github.com/jabir-zheng/TCD' target='_blank'>Github Repo</a></p>"


default_prompt = "Painting of the orange cat Otto von Garfield, Count of Bismarck-Schönhausen, Duke of Lauenburg, Minister-President of Prussia. Depicted wearing a Prussian Pickelhaube and eating his favorite meal - lasagna."
examples = [
    [
        "Beautiful woman, bubblegum pink, lemon yellow, minty blue, futuristic, high-detail, epic composition, watercolor.",
        4
    ],
    [
        "Beautiful man, bubblegum pink, lemon yellow, minty blue, futuristic, high-detail, epic composition, watercolor.",
        8
    ],
    [
        "Painting of the orange cat Otto von Garfield, Count of Bismarck-Schönhausen, Duke of Lauenburg, Minister-President of Prussia. Depicted wearing a Prussian Pickelhaube and eating his favorite meal - lasagna.",
        16
    ],
    [
        "closeup portrait of 1 Persian princess, royal clothing, makeup, jewelry, wind-blown long hair, symmetric, desert, sands, dusty and foggy, sand storm, winds bokeh, depth of field, centered.",
        16
    ],
]

outputs = gr.Label(label='Generated Images')

with gr.Blocks() as demo:
    gr.Markdown(f'# {title}\n### {description}')
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label='Prompt', value=default_prompt)
            num_inference_steps = gr.Slider(
                label='Inference steps',
                minimum=4,
                maximum=16,
                value=4,
                step=1,
            )
            
            with gr.Accordion("Advanced Options", visible=False):
                with gr.Row():
                    with gr.Column():
                        seed = gr.Number(label="Random Seed", value=-1)
                    with gr.Column():
                        eta = gr.Slider(
                                label='Gamma',
                                minimum=0.,
                                maximum=1.,
                                value=0.3,
                                step=0.1,
                            )

            with gr.Row():
                clear = gr.ClearButton(
                    components=[prompt, num_inference_steps, seed, eta])
                submit = gr.Button(value='Submit')

            examples = gr.Examples(
                label="Quick Examples",
                examples=examples,
                inputs=[prompt, num_inference_steps, 0, 0.3],
                outputs="outputs", 
                cache_examples=False
            )

        with gr.Column():
            outputs = gr.Image(label='Generated Images')

    gr.Markdown(f'{article}')

    submit.click(
        fn=inference,
        inputs=[prompt, num_inference_steps, seed, eta],
        outputs=outputs,
    )

demo.launch()
