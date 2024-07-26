import torch
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPImageProcessor
from PIL import Image
import numpy as np
import cv2

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the stable diffusion inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Load the image processor
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load the image
image_path = "/content/assignment.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Add 128 pixels to each side
border_size = 128
image_with_border = Image.new(
    "RGB",
    (image.width + 2 * border_size, image.height + 2 * border_size),
    (255, 255, 255)
)
image_with_border.paste(image, (border_size, border_size))

# Create a mask for the border
mask = np.zeros((image_with_border.height, image_with_border.width), dtype=np.uint8)
mask[:border_size, :] = 1  # Top border
mask[-border_size:, :] = 1  # Bottom border
mask[:, :border_size] = 1  # Left border
mask[:, -border_size:] = 1  # Right border

# Convert the mask to PIL Image
mask_image = Image.fromarray(mask * 255).convert("L")

# Refined prompt to ensure a seamless extension
prompt = (
    "Extend this image by adding natural and seamless extensions to the current scene, "
    "matching the textures, patterns, and colors of the existing image without adding any artificial elements."
)

# Perform inpainting
outpainted_image = pipe(prompt=prompt, image=image_with_border, mask_image=mask_image, guidance_scale=7.5).images[0]

# Display the outpainted image
outpainted_image.show()

# Save the outpainted image
outpainted_image.save("/content/outpainted_image.png")  # Adjust the path as needed
