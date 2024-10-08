import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
from pipelines.models import TextToImageRequest
from torch import Generator
from DeepCache import DeepCacheSDHelper
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)
import packaging.version

from diffusers import UNet2DConditionModel, LCMScheduler
from diffusers import DPMSolverMultistepScheduler
from tgate import TgateSDXLLoader, TgateSDXLDeepCacheLoader

gate_step = 9
inference_step = 9
sp_interval = 1
fi_interval = 0
warm_up = 0


def load_pipeline() -> StableDiffusionXLPipeline:
    vae = AutoencoderTiny.from_pretrained(
    'madebyollin/taesdxl',
    use_safetensors=True,
    torch_dtype=torch.float16,
    ).to('cuda')

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "./models/newdream-sdxl-20",
        torch_dtype=torch.float16,
        use_safetensors=True, 
        local_files_only=True,
        vae=vae,
    )

    pipeline = TgateSDXLLoader(
        pipeline,
        gate_step=gate_step,
        sp_interval=sp_interval,
        fi_interval=fi_interval,
        warm_up=warm_up,
        num_inference_steps=inference_step,
    )
    pipeline.to("cuda")
    config = CompilationConfig.Default()

    try:
        import xformers
        config.enable_xformers = True
    except ImportError:
        print('xformers not installed, skip')
    try:
        import triton
        config.enable_triton = True
    except ImportError:
        print('Triton not installed, skip')

    # config.enable_cuda_graph = True
    pipeline  = compile(pipeline, config)

    for _ in range(6):
        pipeline(prompt="", num_inference_steps=inference_step,)
    
    return pipeline

def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    image = pipeline.tgate(
        request.prompt,
        gate_step=gate_step,
        num_inference_steps=inference_step,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
    ).images[0]

    return image
