import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image
import os
from model import Generator  # use external model

app = Flask(__name__, static_folder='static')

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Generator ---
z_dim = 100
num_classes = 10
img_shape = (1, 28, 28)

generator = Generator(z_dim=z_dim, num_classes=num_classes, img_shape=img_shape).to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# --- Image Generation Function ---
def generate_images(digit, n_images=5):
    z = torch.randn(n_images, z_dim).to(device)
    labels = torch.full((n_images,), digit, dtype=torch.long).to(device)

    with torch.no_grad():
        fake_imgs = generator(z, labels)
        fake_imgs = (fake_imgs + 1) / 2  # Convert [-1, 1] to [0, 1]

    paths = []
    os.makedirs("static/generated", exist_ok=True)



    for i, img_tensor in enumerate(fake_imgs):
        img_pil = transforms.ToPILImage()(img_tensor.cpu())
        path = f"static/generated/digit_{digit}_{i}.png"
        img_pil.save(path)
        paths.append(f"/static/generated/digit_{digit}_{i}.png")


    return paths

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    digit = int(request.json["digit"])
    image_paths = generate_images(digit)
    return jsonify({"images": image_paths})

if __name__ == "__main__":
    app.run(debug=True)
