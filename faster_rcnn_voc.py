from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.datasets import VOCDetection
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm


VOC_CLASSES = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
}

ID_TO_CLASS = {class_id: name for name, class_id in VOC_CLASSES.items()}
NUM_CLASSES = len(VOC_CLASSES) + 1


def collate_detection_batch(batch: List[Tuple[torch.Tensor, dict]]):
    return tuple(zip(*batch))


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    download: bool,
) -> Tuple[DataLoader, DataLoader, VOCDetection]:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = VOCDetection(
        root=str(data_dir),
        year="2007",
        image_set="train",
        download=download,
        transform=transform,
    )
    test_dataset = VOCDetection(
        root=str(data_dir),
        year="2007",
        image_set="test",
        download=download,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_detection_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_detection_batch,
    )

    return train_loader, test_loader, test_dataset


def build_model(full_finetune: bool) -> torch.nn.Module:
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    if not full_finetune:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.roi_heads.box_predictor.parameters():
            parameter.requires_grad = True

    return model


def annotation_objects(raw_target: dict) -> List[dict]:
    objects = raw_target["annotation"].get("object", [])
    if isinstance(objects, dict):
        return [objects]
    return list(objects)


def convert_target(raw_target: dict, device: Optional[torch.device] = None):
    boxes = []
    labels = []

    for obj in annotation_objects(raw_target):
        box = obj["bndbox"]
        boxes.append(
            [
                int(box["xmin"]),
                int(box["ymin"]),
                int(box["xmax"]),
                int(box["ymax"]),
            ]
        )
        labels.append(VOC_CLASSES[obj["name"]])

    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }

    if device is not None:
        target = {key: value.to(device) for key, value in target.items()}

    return target


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, raw_targets in tqdm(dataloader, desc="train", leave=False):
        images = [image.to(device) for image in images]
        targets = [convert_target(target, device) for target in raw_targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += float(losses.item())

    return total_loss / max(len(dataloader), 1)


def validate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    metric = MeanAveragePrecision(
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )

    model.eval()
    with torch.no_grad():
        for images, raw_targets in tqdm(dataloader, desc="validate", leave=False):
            images = [image.to(device) for image in images]
            predictions = model(images)

            processed_predictions = [
                {
                    "boxes": prediction["boxes"].cpu(),
                    "scores": prediction["scores"].cpu(),
                    "labels": prediction["labels"].cpu(),
                }
                for prediction in predictions
            ]
            processed_targets = [
                {
                    "boxes": convert_target(target)["boxes"].cpu(),
                    "labels": convert_target(target)["labels"].cpu(),
                }
                for target in raw_targets
            ]

            metric.update(processed_predictions, processed_targets)

    return metric.compute()


def save_prediction_preview(
    model: torch.nn.Module,
    dataset: VOCDetection,
    device: torch.device,
    output_dir: Path,
    indices: List[int],
    score_threshold: float,
) -> Path:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    fig, axes = plt.subplots(1, len(indices), figsize=(8 * len(indices), 8))
    if len(indices) == 1:
        axes = [axes]

    for axis, index in zip(axes, indices):
        image, raw_target = dataset[index]
        target = convert_target(raw_target)

        with torch.no_grad():
            prediction = model([image.to(device)])[0]

        axis.imshow(F.to_pil_image(image))

        for box, label in zip(target["boxes"], target["labels"]):
            xmin, ymin, xmax, ymax = box.tolist()
            width, height = xmax - xmin, ymax - ymin
            axis.add_patch(
                patches.Rectangle(
                    (xmin, ymin),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
            )
            axis.text(xmin, ymin - 5, ID_TO_CLASS[int(label)], color="red", fontsize=8)

        keep = prediction["scores"].cpu() >= score_threshold
        for box, label in zip(prediction["boxes"].cpu()[keep], prediction["labels"].cpu()[keep]):
            xmin, ymin, xmax, ymax = box.tolist()
            width, height = xmax - xmin, ymax - ymin
            axis.add_patch(
                patches.Rectangle(
                    (xmin, ymin),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="green",
                    facecolor="none",
                )
            )
            axis.text(xmin, ymin - 5, ID_TO_CLASS.get(int(label), str(int(label))), color="green", fontsize=8)

        axis.set_title(f"Image {index}")
        axis.axis("off")

    output_path = output_dir / "predictions.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on Pascal VOC 2007.")
    parser.add_argument("--data-dir", type=Path, default=Path("VOC_data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--preview-indices", type=int, nargs="+", default=[2777, 4742, 777])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and "
            "make sure your NVIDIA driver is working."
        )
    device = torch.device("cuda")

    train_loader, test_loader, test_dataset = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )

    model = build_model(full_finetune=args.full_finetune).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0

    if not args.skip_training:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            print(f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f}")

            if not args.skip_validation:
                results = validate(model, test_loader, device)
                current_map = float(results["map"])
                print(f"Epoch {epoch}/{args.epochs} - mAP@0.5:0.95: {current_map:.4f}")

                if current_map > best_map:
                    best_map = current_map
                    torch.save(model.state_dict(), args.checkpoint_dir / "best_model.pt")

            torch.save(model.state_dict(), args.checkpoint_dir / "last_model.pt")

    if args.visualize:
        output_path = save_prediction_preview(
            model=model,
            dataset=test_dataset,
            device=device,
            output_dir=args.output_dir,
            indices=args.preview_indices,
            score_threshold=args.score_threshold,
        )
        print(f"Saved prediction preview to {output_path}")


if __name__ == "__main__":
    main()
