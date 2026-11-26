import json
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageTk
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "saved"
    / "training"
    / "brain-tumor-automl-resnet18-resnet34-resnet50-resnet101-resnet152-squeezenet-efficientnetb0-freeze-optimizer-epoch-v2-trial-18"
    / "1120_164450"
    / "model_best.pth"
)
NORMALIZATION_PATH = PROJECT_ROOT / "processed_scan_images_mri" / "normalization.json"
CLASS_NAMES_BY_FOLDER = {
    "1": "Meningioma",
    "2": "Glioma",
    "3": "Pituitary tumor",
}
IMAGE_SIZE = (224, 224)
BACKGROUND_THRESHOLD = 8
CROP_MARGIN_FRACTION = 0.05


def load_model(device):
    from model.model import EfficientNetB0

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "The best AutoML checkpoint was not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            "Make sure the saved/training folder is present."
        )
    if not NORMALIZATION_PATH.is_file():
        raise FileNotFoundError(
            f"Training normalization values were not found:\n{NORMALIZATION_PATH}"
        )

    with NORMALIZATION_PATH.open("r", encoding="utf-8") as file:
        normalization = json.load(file)
    mean = normalization.get("mean")
    std = normalization.get("std")
    if (
        not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != 3
        or len(std) != 3
        or any(value <= 0 for value in std)
    ):
        raise ValueError(f"Invalid image normalization values in {NORMALIZATION_PATH}.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no valid state_dict: {CHECKPOINT_PATH}")

    model = EfficientNetB0(num_classes=len(CLASS_NAMES_BY_FOLDER), use_pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, image_transform


def preprocess_image(image_path):
    with Image.open(image_path) as source_image:
        rgba = ImageOps.exif_transpose(source_image).convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
        gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(
        gray, BACKGROUND_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if width * height >= 0.15 * gray.shape[0] * gray.shape[1]:
            margin = max(4, int(CROP_MARGIN_FRACTION * max(width, height)))
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(gray.shape[1], x + width + margin)
            y1 = min(gray.shape[0], y + height + margin)
            gray = gray[y0:y1, x0:x1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    image = Image.fromarray(gray)
    image.thumbnail(IMAGE_SIZE, Image.Resampling.BILINEAR)
    canvas = Image.new("L", IMAGE_SIZE, 0)
    offset = ((IMAGE_SIZE[0] - image.width) // 2, (IMAGE_SIZE[1] - image.height) // 2)
    canvas.paste(image, offset)
    gray_array = np.asarray(canvas)
    rgb_array = np.repeat(gray_array[:, :, np.newaxis], 3, axis=2)
    return Image.fromarray(rgb_array, mode="RGB")


class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain MRI Tumour Classifier")
        self.root.geometry("700x760")
        self.root.minsize(560, 620)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None
        self.image_transform = None
        self.preview_image = None
        self.preview_photo = None

        container = ttk.Frame(root, padding=24)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            container,
            text="Brain MRI Tumour Classifier",
            font=("TkDefaultFont", 20, "bold"),
        ).pack(pady=(0, 8))
        ttk.Label(
            container,
            text="Choose an MRI image to see the model's predicted tumour class.",
            wraplength=620,
        ).pack(pady=(0, 18))

        ttk.Button(
            container,
            text="Upload MRI image",
            command=self.select_image,
        ).pack(pady=(0, 16))

        self.filename_label = ttk.Label(container, text="No image selected")
        self.filename_label.pack(pady=(0, 10))

        self.preview_canvas = tk.Canvas(
            container,
            highlightthickness=0,
            borderwidth=0,
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, pady=8)
        self.preview_canvas.bind("<Configure>", self.render_preview)

        self.prediction_label = ttk.Label(
            container,
            text="",
            font=("TkDefaultFont", 17, "bold"),
            justify=tk.CENTER,
        )
        self.prediction_label.pack(pady=(12, 4))

        self.probabilities_label = ttk.Label(
            container,
            text="",
            justify=tk.CENTER,
        )
        self.probabilities_label.pack(pady=(0, 12))

        ttk.Label(
            container,
            text=f"Running on {self.device}. This model output is not a medical diagnosis.",
            wraplength=620,
            justify=tk.CENTER,
        ).pack(pady=(4, 0))

    def select_image(self):
        image_path = filedialog.askopenfilename(
            title="Select a brain MRI image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*"),
            ],
        )
        if not image_path:
            return

        try:
            image = preprocess_image(image_path)
            self.show_preview(image)
            self.filename_label.config(text=Path(image_path).name)
            self.predict(image)
        except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
            self.prediction_label.config(text="")
            self.probabilities_label.config(text="")
            messagebox.showerror("Could not make a prediction", str(error), parent=self.root)

    def show_preview(self, image):
        self.preview_image = image
        self.render_preview()

    def render_preview(self, _event=None):
        if self.preview_image is None:
            return

        canvas_width = (
            _event.width if _event is not None else self.preview_canvas.winfo_width()
        )
        canvas_height = (
            _event.height if _event is not None else self.preview_canvas.winfo_height()
        )
        if canvas_width <= 1 or canvas_height <= 1:
            return

        max_width = max(1, min(440, canvas_width - 16))
        max_height = max(1, min(440, canvas_height - 16))
        preview = ImageOps.contain(
            self.preview_image,
            (max_width, max_height),
        )
        scale = min(
            max_width / preview.width,
            max_height / preview.height,
        )
        if scale > 1:
            preview = preview.resize(
                (int(preview.width * scale), int(preview.height * scale)),
                Image.Resampling.LANCZOS,
            )

        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_canvas.delete("preview")
        self.preview_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.preview_photo,
            anchor=tk.CENTER,
            tags="preview",
        )

    def predict(self, image):
        if self.model is None or self.image_transform is None:
            self.prediction_label.config(text="Loading model…")
            self.root.update_idletasks()
            self.model, self.image_transform = load_model(self.device)

        input_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(input_tensor), dim=1)[0].cpu()

        class_folders = sorted(CLASS_NAMES_BY_FOLDER)
        if probabilities.numel() != len(class_folders):
            raise ValueError(
                f"Expected {len(class_folders)} output classes, "
                f"received {probabilities.numel()}."
            )
        predicted_index = int(probabilities.argmax().item())
        predicted_name = CLASS_NAMES_BY_FOLDER[class_folders[predicted_index]]

        self.prediction_label.config(
            text=f"Prediction: {predicted_name}\n"
            f"Softmax score: {probabilities[predicted_index].item():.1%}"
        )
        probability_lines = [
            f"{CLASS_NAMES_BY_FOLDER[folder]}: {probabilities[index].item():.1%}"
            for index, folder in enumerate(class_folders)
        ]
        self.probabilities_label.config(text="\n".join(probability_lines))


def main():
    root = tk.Tk()
    PredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
