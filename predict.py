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
        seed: int = Input(default=42, description="Random seed"),
        num_inference_steps: int = Input(default=50, description="Sampling steps")
    ) -> Path:
        result = infer(self.model, prompt=prompt, seed=seed, num_inference_steps=num_inference_steps)
        image: Image.Image = result["images"][0]

        out_path = f"/tmp/{uuid.uuid4().hex}.png"
        image.save(out_path)
        return Path(out_path)
