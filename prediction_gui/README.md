# Brain MRI prediction GUI

This standalone desktop GUI uses the best saved AutoML EfficientNetB0 checkpoint
to predict whether an uploaded MRI image belongs to the meningioma, glioma, or
pituitary-tumour class.

## Run

From the project root, activate the project's Python environment and run:

```bash
python prediction_gui/app.py
```

The app uses the project's existing PyTorch, Torchvision, Pillow, OpenCV, and
NumPy dependencies; no additional GUI package is required. It expects the best
AutoML checkpoint at:

```text
saved/training/brain-tumor-automl-resnet18-resnet34-resnet50-resnet101-resnet152-squeezenet-efficientnetb0-freeze-optimizer-epoch-v2-trial-18/1120_164450/model_best.pth
```

Uploaded images go through the project's preprocessing steps (grayscale
conversion, dark-background crop, CLAHE, and aspect-preserving 224 × 224
resize/padding), followed by the same normalization used during evaluation.
The model's prediction is for research/demo use and is not a medical diagnosis.
