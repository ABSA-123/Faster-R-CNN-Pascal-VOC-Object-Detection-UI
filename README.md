# Faster R-CNN Pascal VOC Object Detection UI

This project is a Python object detection application that trains and runs a
Faster R-CNN MobileNetV3 model on the Pascal VOC 2007 dataset. It includes a
Streamlit user interface where users can upload an image, run detection, and
view the detected objects with confidence scores and bounding boxes.

## Features

- Train Faster R-CNN on Pascal VOC 2007.
- Run object detection on uploaded JPG or PNG images.
- Dark themed Streamlit UI.
- Confidence threshold slider.
- Progress bar while detection is running.
- Dedicated results section.
- Annotated image output with bounding boxes.
- Recognition summary with object names, confidence values, and box positions.
- Pascal VOC object class list shown in the UI.
- CUDA GPU execution for local training and inference.

## Supported Object Classes

The Pascal VOC model recognizes these 20 object categories:

- aeroplane
- bicycle
- bird
- boat
- bottle
- bus
- car
- cat
- chair
- cow
- diningtable
- dog
- horse
- motorbike
- person
- pottedplant
- sheep
- sofa
- train
- tvmonitor

## Project Structure

```text
fasterrcnn_voc_project/
|-- app.py                # Streamlit user interface
|-- faster_rcnn_voc.py    # Training, validation, and model utilities
|-- requirements.txt      # Python dependencies
|-- README.md             # Project documentation
|-- checkpoints/          # Trained model checkpoints, ignored by Git
`-- VOC_data/             # Pascal VOC dataset, ignored by Git
```

## Requirements

- Python 3.8+
- NVIDIA GPU recommended
- CUDA-enabled PyTorch build
- Pascal VOC 2007 dataset

Install dependencies:

```powershell
pip install -r requirements.txt
```

If you use the included virtual environment locally, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## CUDA Setup

This project is configured to use CUDA. If your environment has CPU-only
PyTorch, reinstall PyTorch with CUDA support:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Train the Model

Train Faster R-CNN on Pascal VOC:

```powershell
.\.venv\Scripts\python.exe faster_rcnn_voc.py --epochs 10 --batch-size 1
```

Checkpoints are saved in:

```text
checkpoints/
```

The UI expects one of these files:

```text
checkpoints/best_model.pt
checkpoints/last_model.pt
```

## Run the UI

Start the Streamlit app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If port `8501` is already busy:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8601
```

Then open the local URL shown in the terminal.

## How to Use

1. Run the Streamlit app.
2. Upload a JPG or PNG image.
3. Adjust the confidence threshold if needed.
4. Confirm that a trained checkpoint is loaded.
5. Click **Run Detection**.
6. Wait for the progress bar to finish.
7. View the detected objects, annotated image, confidence scores, and result
   table in the Results section.

## Notes for GitHub

The dataset, virtual environment, trained checkpoints, reports, and generated
artifacts are intentionally ignored by Git. This keeps the repository small and
focused on source code.

Ignored local files include:

- `.venv/`
- `VOC_data/`
- `checkpoints/`
- `outputs/`
- `*.pt`
- `*.pth`
- `*.pdf`
- `*.docx`
- log files

## Assignment Context

This project was built as a multimedia-based system prototype for object
detection and recognition. It demonstrates image processing and deep learning
through a working application with a user interface, model inference, testing
screenshots, and documented execution steps.
