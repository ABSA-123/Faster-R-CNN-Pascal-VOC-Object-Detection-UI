# Faster R-CNN Pascal VOC

Clean Python project for training and demonstrating a Faster R-CNN MobileNetV3
object detector on Pascal VOC 2007. The Streamlit UI lets users upload images,
run object detection, view bounding boxes, and see the list of detectable VOC
objects.

## Setup

```bash
pip install -r requirements.txt
```

## Run the UI

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, upload a JPG or PNG image, adjust the
confidence threshold, and review the annotated image plus the detection summary.

The UI loads the trained Pascal VOC detector from `checkpoints/best_model.pt` or
`checkpoints/last_model.pt`. Train the model first if neither checkpoint exists.

## Pascal VOC Objects

When a trained VOC checkpoint is loaded, the detector recognizes these 20 object
categories:

```text
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
train, tvmonitor
```

## Train

```bash
python faster_rcnn_voc.py --epochs 10 --batch-size 1
```

Checkpoints are saved in `checkpoints/`.

## Optional Prediction Preview

```bash
python faster_rcnn_voc.py --visualize --skip-training
```
