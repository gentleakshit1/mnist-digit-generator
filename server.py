import os
from flask import Flask

app = Flask(__name__)

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    digit = request.json["digit"]
    
    # Return 5 dummy images (must exist in static/generated/)
    images = [
        f"/static/generated/1.png",
        f"/static/generated/2.png",
        f"/static/generated/3.png",
        f"/static/generated/4.png",
        f"/static/generated/5.png",
    ]
    return jsonify({"images": images})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
