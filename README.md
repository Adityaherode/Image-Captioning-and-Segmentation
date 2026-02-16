# Image Captioning + Segmentation (TensorFlow)

1. Create a virtual env:
   python -m venv .venv
   source .venv/bin/activate   # (Linux / macOS)
   .\.venv\Scripts\activate    # (Windows)

2. Install dependencies:
   pip install - import numpy

3. Run the app:
   streamlit run app.py

# Notes:
- Training full models on COCO requires a GPU and many hours. The scripts include training hooks and a small quick-mode (use a few images) for experimentation.
- For segmentation demo you can use the built-in GrabCut fallback for quick results without training.

