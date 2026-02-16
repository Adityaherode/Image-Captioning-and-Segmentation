# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
from captioning.extract_features import build_feature_extractor, extract_feature_from_path
from captioning.model import CaptioningModel, SimpleTokenizer
from segmentation.simple_seg import grabcut_segmentation

st.set_page_config(page_title="Image Caption + Segment Demo")

st.title("Image Captioning + Segmentation Demo")

uploaded = st.file_uploader("Upload an image", type=['png','jpg','jpeg'])
if uploaded is not None:
    # read and show
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)
    # save temporarily
    with open("temp_input.jpg", "wb") as f:
        f.write(uploaded.getbuffer())

    st.write("Running segmentation (GrabCut fallback)...")
    seg_out, mask = grabcut_segmentation("temp_input.jpg")
    st.image(seg_out[:,:,::-1], caption="Segmentation output", use_column_width=True)

    st.write("Running captioning feature extraction...")
    feat_model = build_feature_extractor()
    feat = extract_feature_from_path(feat_model, "temp_input.jpg")
    # dummy tokenizer for demo — real app should load a trained tokenizer
    dummy_index = {"startseq":1, "a":2, "man":3, "with":4, "dog":5, "endseq":6}
    tokenizer = SimpleTokenizer(dummy_index)
    cap_model = CaptioningModel(vocab_size=100, max_length=20)
    st.write("Caption (untrained demo model):")
    st.text(cap_model.generate_caption(feat, tokenizer))
    st.info("Note: caption model is not trained in this demo. To get meaningful captions, train the captioning model and load weights.")
    