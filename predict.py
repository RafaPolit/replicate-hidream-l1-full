from cog import BasePredictor, Input, Path
import mmengine
from mmagic.apis import init_model, infer
import torch
import os
import uuid
from PIL import Image

CONFIG_URL = "https://huggingface.co/fffiloni/hidream-l1-full-replicate/resolve/main/hidream-l1-full_512x512_fp16.yaml"
CHECKPOINT_URL = "https://huggingface.co/fffiloni/hidream-l1-full-replicate/resolve/main/hidream-l1-full_512x512_fp16.pth"

CONFIG_PATH = "hidream-l1-full.yaml"
CHECKPOINT_PATH = "hidream-l1-full.pth"


class Predictor(BasePredictor):
    def setup(self):
        if not os.path.exists(CONFIG_PATH):
            os.system(f"wget {CONFIG_URL} -O {CONFIG_PATH}")
        if not os.path.exists(CHECKPOINT_PATH):
            os.system(f"wget {CHECKPOINT_URL} -O {CHECKPOINT_PATH}")

        self.model = init_model(CONFIG_PATH, CHECKPOINT_PATH, device="cuda")

    def predict(
        self,
        prompt: str = Input(description="The input text prompt"),
        negative_prompt: str = Input(default="", description="Text to avoid in the output"),
        height: int = Input(default=512, ge=64, le=1024, description="Height of output image"),
        width: int = Input(default=512, ge=64, le=1024, description="Width of output image"),
        guidance_scale: float = Input(default=7.5, ge=0.0, le=20.0, description="Classifier-free guidance scale"),
        num_inference_steps: int = Input(default=50, ge=1, le=100, description="Sampling steps"),
        seed: int = Input(default=42, description="Random seed for reproducibility"),
        sampler: str = Input(default="ddim", choices=["ddim", "dpm", "euler", "pndm"], description="Sampling algorithm")
    ) -> Path:
        result = infer(
            self.model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            sampler=sampler
        )
        image: Image.Image = result["images"][0]
        out_path = f"/tmp/{uuid.uuid4().hex}.png"
        image.save(out_path)
        return Path(out_path)
