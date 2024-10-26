import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline
from pipelines.models import TextToImageRequest
from torch import Generator
from onediffx import compile_pipe

def load_pipeline(pipeline=None) -> StableDiffusionXLPipeline:
    if not pipeline:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "./models/newdream-sdxl-20",
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")
    pipeline = compile_pipe(pipeline)
    for _ in range(3):
        pipeline(prompt="photo of a cat", num_inference_steps=20)

    return pipeline

def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
  if step_index == int(pipe.num_timesteps * 0.5):
    callback_kwargs['prompt_embeds'] = callback_kwargs['prompt_embeds'].chunk(2)[-1]
    callback_kwargs['add_text_embeds'] = callback_kwargs['add_text_embeds'].chunk(2)[-1]
    callback_kwargs['add_time_ids'] = callback_kwargs['add_time_ids'].chunk(2)[-1]
    pipe._guidance_scale = 0.0

  return callback_kwargs

def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    if request.seed is None:
        generator = None
    else:
        generator = Generator(pipeline.device).manual_seed(request.seed)

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
        num_inference_steps=20,
        callback_on_step_end=callback_dynamic_cfg,
        callback_on_step_end_tensor_inputs=['prompt_embeds', 'add_text_embeds', 'add_time_ids'],
    ).images[0]