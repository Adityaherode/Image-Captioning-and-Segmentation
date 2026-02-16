# Image Captioning + Segmentation (TensorFlow)

## Quick setup
1. Create a virtual env:
   python -m venv .venv
   source .venv/bin/activate   # (Linux / macOS)
   .\.venv\Scripts\activate    # (Windows)

2. Install dependencies:
   pip install -r requirements.txt

3. (Optional) Download datasets:
   - Flickr8k captions/images or MSCOCO/Pascal VOC for full training (see utils/data_utils.py).

4. Run the demo app:
   streamlit run app.py

## Notes
- Training full models on COCO requires a GPU and many hours. The scripts include training hooks and a small quick-mode (use a few images) for experimentation.
- For segmentation demo you can use the built-in GrabCut fallback for quick results without training.