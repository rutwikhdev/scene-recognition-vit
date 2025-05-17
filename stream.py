import numpy as np

import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from transformers import AutoImageProcessor

CLASS_LABELS = [
    "airport_terminal",
    "amphitheatre",
    "amusement_park",
    "art_gallery",
    "bakery_shop",
    "bar",
    "bookstore",
    "botanical_garden",
    "bridge",
    "bus interior",
    "butchers shop",
    "campsite",
    "classroom",
    "coffee_shop",
    "construction_site",
    "courtyard",
    "driveway",
    "fire_station",
    "fountain",
    "gas_station",
    "harbour",
    "highway",
    "kindergarten_classroom",
    "lobby",
    "market_outdoor",
    "museum",
    "office",
    "parking_lot",
    "phone_booth",
    "playground",
    "railroad_track",
    "restaurant",
    "river",
    "shed",
    "staircase",
    "supermarket",
    "swomming_pool_outdoor",
    "track",
    "valley",
    "yard",
]

MODEL_OPTIONS = {
    "DINOv2": ("dinov2", "./dino.pth", "facebook/dinov2-base"),
    "ViT": ("vit_base", "./vit.pth", "google/vit-base-patch16-224"),
    "Swin": ("swin", "./swin.pth", "microsoft/swin-tiny-patch4-window7-224"),
}

selected_model_name = st.selectbox("Select a Model", list(MODEL_OPTIONS.keys()))
selected_arch, selected_path, img_processor = MODEL_OPTIONS[selected_model_name]


@st.cache_resource
def load_model(arch, path):
    from src.models import load_vit_model

    model = load_vit_model(arch).to("cuda")
    state_dict = torch.load(path, map_location="cuda")
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


model = load_model(selected_arch, selected_path)
feature_extractor = AutoImageProcessor.from_pretrained(img_processor)


def preprocess_image(img):
    return feature_extractor(images=img, return_tensors="pt")


def visualize_attention(img, outputs):
    # Extract and average attention across layers and heads
    attentions = outputs.attentions
    attn = torch.stack(attentions)
    attn = attn[:, 0]
    attn = attn.mean(dim=0)

    num_heads = attn.shape[0]
    num_tokens = attn.shape[-1]

    # Get width/height of the attention map
    num_patches = num_tokens - 1
    attn_size = int(np.sqrt(num_patches))

    # Confirm patch count is square
    assert attn_size**2 == num_patches, "Unexpected number of patches"

    # Prepare image dimensions
    w, h = img.size

    grid_size = int(np.ceil(np.sqrt(num_heads)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 14))

    for i in range(num_heads):
        # Get attention from CLS token to all patches
        cls_attn = attn[i, 0, 1:].reshape(attn_size, attn_size).detach().cpu().numpy()

        # Normalize and upscale to image size
        cls_attn = cls_attn / cls_attn.max()
        cls_attn = cv2.resize(cls_attn, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attn), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(np.array(img.convert("RGB")), 0.6, heatmap, 0.4, 0)

        ax = axes[i // grid_size, i % grid_size]
        ax.imshow(overlay)
        ax.set_title(f"Head {i + 1}")
        ax.axis("off")

    # Hide unused subplots
    for i in range(num_heads, grid_size * grid_size):
        axes[i // grid_size, i % grid_size].axis("off")

    st.pyplot(fig)


st.title("Vision Transformer Attention Visualizer")

st.markdown(
    """
    <style>
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 90% !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    inputs = preprocess_image(image).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    logits = outputs.logits
    prediction = logits.argmax(dim=-1).item()
    st.write(f"**Predicted Class:** {CLASS_LABELS[prediction]}")

    st.subheader("Attention Heads Visualization")
    visualize_attention(image, outputs)
