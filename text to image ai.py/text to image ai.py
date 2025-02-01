from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

pipeline=StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline.to("cuda")
prompt ="nature image"
image = pipeline(prompt, num_inference_steps=50).images[0]

plt.imshow(image)
plt.axis("off")
plt.show()