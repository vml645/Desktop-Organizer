# Author: Venkat
# Date: 2021-09-26
# Desktop Screenshot Organizer

import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
import clip
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

screenshots_dir = "/Users/venkat/Desktop"

# Load the screenshots
image_files = [f for f in os.listdir(screenshots_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

# Extract the embeddings
image_embeddings = []
for image_file in image_files:
    image_path = os.path.join(screenshots_dir, image_file)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_embeddings.append(image_features.cpu().numpy())

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(np.array(image_embeddings).reshape(len(image_embeddings), -1))

# Initialize the BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

for i in range(num_clusters):
    cluster_images = [image_files[j] for j in range(len(image_files)) if cluster_labels[j] == i]
    if len(cluster_images) > 0:
        image_path = os.path.join(screenshots_dir, cluster_images[0])
        image = Image.open(image_path)

        # Generate caption using the BLIP model
        inputs = blip_processor(image, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)

        # Create a readable folder name based on the caption
        words = caption.split()[:3]  # Use the first three words of the caption
        folder_name = " ".join(word.capitalize() for word in words)
        folder_path = os.path.join(screenshots_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for image_file in cluster_images:
            src_path = os.path.join(screenshots_dir, image_file)
            dst_path = os.path.join(folder_path, image_file)
            shutil.move(src_path, dst_path)