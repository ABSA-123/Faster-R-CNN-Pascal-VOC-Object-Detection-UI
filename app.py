from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont

from faster_rcnn_voc import ID_TO_CLASS, VOC_CLASSES, build_model


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
MAX_INFERENCE_SIDE = 1000
VOC_CHECKPOINTS = [
    CHECKPOINT_DIR / "best_model.pt",
    CHECKPOINT_DIR / "last_model.pt",
]

PALETTE = [
    "#60a5fa",
    "#f87171",
    "#34d399",
    "#c084fc",
    "#fb923c",
    "#22d3ee",
    "#f43f5e",
    "#818cf8",
    "#4ade80",
    "#facc15",
]


def available_voc_checkpoint() -> Path | None:
    for checkpoint_path in VOC_CHECKPOINTS:
        if checkpoint_path.exists():
            return checkpoint_path
    return None


@st.cache_resource(show_spinner=False)
def load_detector(checkpoint_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and "
            "make sure your NVIDIA driver is working."
        )
    device = torch.device("cuda")

    model = build_model(full_finetune=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, device, ID_TO_CLASS, list(VOC_CLASSES.keys())


def predict_image(
    model: torch.nn.Module,
    device: torch.device,
    image: Image.Image,
    score_threshold: float,
) -> List[dict]:
    tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([tensor])[0]

    detections = []
    for box, label, score in zip(
        prediction["boxes"].detach().cpu(),
        prediction["labels"].detach().cpu(),
        prediction["scores"].detach().cpu(),
    ):
        confidence = float(score.item())
        if confidence < score_threshold:
            continue
        detections.append(
            {
                "box": [float(value) for value in box.tolist()],
                "label_id": int(label.item()),
                "score": confidence,
            }
        )

    return detections


def resize_for_inference(image: Image.Image) -> Tuple[Image.Image, bool]:
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= MAX_INFERENCE_SIDE:
        return image, False

    scale = MAX_INFERENCE_SIDE / longest_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), True


def fit_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
) -> Tuple[str, ImageFont.ImageFont]:
    font = ImageFont.load_default()
    display_text = text
    while (
        len(display_text) > 4
        and draw.textbbox((0, 0), display_text, font=font)[2] > max_width
    ):
        display_text = f"{display_text[:-4]}..."
    return display_text, font


def draw_detections(
    image: Image.Image,
    detections: List[dict],
    label_names: Dict[int, str],
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    for index, detection in enumerate(detections):
        xmin, ymin, xmax, ymax = detection["box"]
        xmin = max(0, min(width - 1, xmin))
        ymin = max(0, min(height - 1, ymin))
        xmax = max(0, min(width - 1, xmax))
        ymax = max(0, min(height - 1, ymax))

        label = label_names.get(detection["label_id"], str(detection["label_id"]))
        label = label.replace("_", " ")
        text = f"{label} {detection['score']:.0%}"
        color = PALETTE[index % len(PALETTE)]
        text, font = fit_label(draw, text, max_width=max(32, int(xmax - xmin) - 8))

        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=4)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_y = max(0, ymin - text_height - 8)
        draw.rectangle(
            (xmin, label_y, min(width, xmin + text_width + 10), label_y + text_height + 8),
            fill=color,
        )
        draw.text((xmin + 5, label_y + 3), text, fill="white", font=font)

    return annotated


def detection_rows(detections: List[dict], label_names: Dict[int, str]) -> List[dict]:
    rows = []
    for detection in detections:
        label = label_names.get(detection["label_id"], str(detection["label_id"]))
        xmin, ymin, xmax, ymax = detection["box"]
        rows.append(
            {
                "Object": label.replace("_", " "),
                "Confidence": f"{detection['score']:.1%}",
                "Box": f"{xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f}",
            }
        )
    return rows


def render_supported_objects(class_names: List[str]) -> None:
    object_list = "\n".join(
        f"- {class_name.replace('_', ' ')}" for class_name in class_names
    )
    st.markdown(object_list)


def main() -> None:
    st.set_page_config(
        page_title="Faster R-CNN Object Detection",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .stApp {
                background: #0b1020;
                color: #e5e7eb;
            }
            .block-container {
                padding-top: 1.75rem;
                padding-bottom: 2rem;
                max-width: 1180px;
            }
            h1, h2, h3 {
                letter-spacing: 0;
                color: #f8fafc;
            }
            p, label, span, div {
                color: inherit;
            }
            [data-testid="stHeader"] {
                background: #070b16;
            }
            [data-testid="stFileUploaderDropzone"] {
                background: #121827;
                border: 1px solid #334155;
                color: #e5e7eb;
            }
            [data-testid="stFileUploaderDropzone"] small,
            [data-testid="stFileUploaderDropzone"] span {
                color: #cbd5e1;
            }
            [data-testid="stSlider"] {
                color: #e5e7eb;
            }
            .stButton > button {
                background: #2563eb;
                border: 1px solid #60a5fa;
                color: #ffffff;
                border-radius: 8px;
                font-weight: 700;
            }
            .stButton > button:hover {
                background: #1d4ed8;
                border-color: #93c5fd;
                color: #ffffff;
            }
            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 6px 10px;
                border: 1px solid #334155;
                border-radius: 999px;
                background: #111827;
                color: #dbeafe;
                font-size: 0.86rem;
            }
            .metric-row {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin: 12px 0 20px;
            }
            .metric-box {
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 12px;
                background: #111827;
            }
            .metric-label {
                font-size: 0.78rem;
                color: #94a3b8;
                margin-bottom: 3px;
            }
            .metric-value {
                font-size: 1.35rem;
                font-weight: 700;
                color: #f8fafc;
            }
            .object-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 8px;
                border-top: 1px solid #334155;
                padding-top: 14px;
                margin-top: 12px;
            }
            .object-chip {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 10px;
                min-height: 38px;
                border: 1px solid #334155;
                border-radius: 8px;
                background: #111827;
                color: #e5e7eb;
                padding: 8px 10px;
                text-transform: capitalize;
                font-size: 0.92rem;
            }
            .object-icon {
                color: #93c5fd;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .stDataFrame {
                border: 1px solid #334155;
                border-radius: 8px;
                overflow: hidden;
            }
            hr {
                border-color: #334155;
            }
            @media (max-width: 720px) {
                .metric-row,
                .object-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    checkpoint_path = available_voc_checkpoint()
    has_checkpoint = checkpoint_path is not None

    st.title("Faster R-CNN Object Detection")
    st.markdown(
        '<span class="status-pill">Model: Pascal VOC Faster R-CNN</span>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Upload Image")
        uploaded_image = st.file_uploader(
            "Choose a JPG or PNG image",
            type=["jpg", "jpeg", "png"],
        )

        score_threshold = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.55,
            step=0.05,
        )

        if not has_checkpoint:
            st.error(
                "No trained Pascal VOC checkpoint was found in checkpoints/. "
                "Train the model first so the app can load best_model.pt or last_model.pt."
            )
        else:
            st.success(f"Using checkpoint: {checkpoint_path.name}")

    with right:
        st.subheader("Objects It Can Detect")
        listed_classes = list(VOC_CLASSES.keys())
        st.write(
            "This project recognizes the Pascal VOC object categories after "
            "loading a trained checkpoint from faster_rcnn_voc.py."
        )
        render_supported_objects(listed_classes)

    if uploaded_image is None:
        st.divider()
        st.header("Results")
        st.info("Upload an image to run object detection.")
        return

    try:
        image = Image.open(uploaded_image).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read this image file: {exc}")
        return

    inference_image, was_resized = resize_for_inference(image)
    st.success(
        f"Image uploaded successfully: {uploaded_image.name} "
        f"({image.size[0]} x {image.size[1]} pixels)."
    )

    preview_a, preview_b = st.columns([1.25, 0.75], gap="large")
    with preview_a:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
    with preview_b:
        st.subheader("Next Step")
        if was_resized:
            st.info(
                "This image is large, so the app will use a resized copy for "
                "faster detection."
            )
        if has_checkpoint:
            st.write("Click the button below to start object detection.")
        else:
            st.write("Train the VOC model before running detection.")
        run_detection = st.button(
            "Run Detection",
            type="primary",
            disabled=not has_checkpoint,
        )

    st.divider()
    st.header("Results")

    if not run_detection:
        if has_checkpoint:
            st.info("Your image is ready. Click Run Detection to see the results here.")
        else:
            st.error(
                "Results cannot run yet because checkpoints/best_model.pt or "
                "checkpoints/last_model.pt is missing."
            )
        return

    progress_text = st.empty()
    progress_bar = st.progress(0)

    try:
        progress_text.info("Loading detector...")
        progress_bar.progress(25)
        model, device, label_names, _supported_classes = load_detector(str(checkpoint_path))

        progress_text.info("Running object detection...")
        progress_bar.progress(65)
        detections = predict_image(model, device, inference_image, score_threshold)

        progress_text.info("Drawing detection results...")
        progress_bar.progress(90)
        annotated_image = draw_detections(inference_image, detections, label_names)

        progress_bar.progress(100)
        progress_text.success("Detection finished.")
    except Exception as exc:
        progress_bar.empty()
        progress_text.empty()
        st.error(f"Detection failed: {exc}")
        st.info(
            "Try restarting Streamlit from the project virtual environment, then "
            "upload a JPG or PNG image again."
        )
        return

    detected_names = [
        label_names.get(detection["label_id"], str(detection["label_id"])).replace("_", " ")
        for detection in detections
    ]
    counts = Counter(detected_names)
    strongest = max((detection["score"] for detection in detections), default=0.0)

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">Detected objects</div>
                <div class="metric-value">{len(detections)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Unique classes</div>
                <div class="metric-value">{len(counts)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Top confidence</div>
                <div class="metric-value">{strongest:.0%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_a, result_b = st.columns([1.25, 0.75], gap="large")
    with result_a:
        st.subheader("Detection Result")
        st.image(annotated_image, use_container_width=True)

    with result_b:
        st.subheader("Recognition Summary")
        if detections:
            summary = ", ".join(f"{count} {name}" for name, count in counts.items())
            st.success(f"The model found: {summary}.")
            st.dataframe(
                detection_rows(detections, label_names),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning(
                "No objects passed the current confidence threshold. Try a lower "
                "threshold or upload a clearer image."
            )


if __name__ == "__main__":
    main()
