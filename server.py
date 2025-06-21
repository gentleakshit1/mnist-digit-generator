import os
import gc
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from model import Generator

app = Flask(__name__, static_folder='static')

# --- Device & Paths ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
num_classes = 10
img_shape = (1, 28, 28)
generator_path = "generator.pth"

# --- Load Generator (only once) ---
generator = Generator(z_dim=z_dim, num_classes=num_classes, img_shape=img_shape).to(device)

try:
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    print("[INFO] Generator model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load generator model: {e}")

# --- Image Generation ---
def generate_images(digit, n_images=5):
    z = torch.randn(n_images, z_dim, dtype=torch.float32).to(device)
    labels = torch.full((n_images,), digit, dtype=torch.long).to(device)

    with torch.no_grad():
        fake_imgs = generator(z, labels)
        fake_imgs = (fake_imgs + 1) / 2  # Convert [-1, 1] to [0, 1]

    # Create directory
    output_dir = os.path.join("static", "generated")
    os.makedirs(output_dir, exist_ok=True)

    # Optional: Clear old images
    for f in os.listdir(output_dir):
        try:
            os.remove(os.path.join(output_dir, f))
        except Exception as e:
            print(f"[WARN] Could not remove old file: {f} - {e}")

    # Save new images
    paths = []
    for i, img_tensor in enumerate(fake_imgs):
        img_pil = transforms.ToPILImage()(img_tensor.cpu())
        filename = f"digit_{digit}_{i}.png"
        path = os.path.join(output_dir, filename)
        img_pil.save(path)
        paths.append(f"/static/generated/{filename}")

    # Clean up
    del fake_imgs, z, labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return paths

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        digit = int(request.json["digit"])
        image_paths = generate_images(digit)
        return jsonify({"images": image_paths})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 if PORT not set
    app.run(host="0.0.0.0", port=port)
