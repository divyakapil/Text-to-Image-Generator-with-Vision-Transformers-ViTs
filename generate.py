from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
                                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "A fantasy landscape with mountains and rivers"
image = pipe(prompt).images[0]
image.save("generated_image.png")