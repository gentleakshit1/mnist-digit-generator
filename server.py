import torch
import torch.nn as nn
import torchvision.utils as vutils
import os
import numpy as np
from model import Generator
from flask import Flask, request, render_template, jsonify
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Load the generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(...)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.to(device)
generator.eval()

# Digit embedding labels: 0–9
def generate_images(digit, n_images=5):
    z = torch.randn(n_images, 100).to(device)  # latent vector
    labels = torch.full((n_images,), digit, dtype=torch.long).to(device)

    with torch.no_grad():
        fake_imgs = generator(z, labels)
        fake_imgs = (fake_imgs + 1) / 2  # [-1,1] → [0,1]

    paths = []
    for i, img_tensor in enumerate(fake_imgs):
        img_pil = transforms.ToPILImage()(img_tensor.cpu())
        filename = f"static/generated/{digit}_{i}.png"
        img_pil.save(filename)
        paths.append(f"/static/generated/{digit}_{i}.png")

    return paths

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    digit = int(request.json["digit"])
    images = generate_images(digit)
    return jsonify({"images": images})
