# Image Outpainting with Stable Diffusion

This project demonstrates how to extend the borders of an image using the Stable Diffusion inpainting model. The goal is to add 128 pixels to each side of the image while ensuring that the new pixels blend seamlessly with the original image.

## Overview

This repository includes a Python script that performs image outpainting using the `diffusers` library for inpainting with Stable Diffusion. The script adds a border around the image, creates a mask for the new border, and then uses the inpainting model to generate extensions that blend with the original image.

## Features

- Seamless extension of images by adding borders
- Utilizes Stable Diffusion for inpainting
- Generates high-quality outpainted images

## Prerequisites

Before running the script, make sure you have the following installed:

- Python 3.x
- `pip` (Python package installer)

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/amanpundhir/Outpainting-Image.git
cd image-outpainting
```

Install the required libraries:

```bash
pip install diffusers transformers accelerate torch opencv-python-headless
```

Place the image you want to outpaint in the directory or adjust the image_path variable in the script to point to your image location.

Execute the script using Python:

```bash
python outpaint.py
```
