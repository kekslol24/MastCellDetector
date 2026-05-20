from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

CLASS_NAMES = ["Atypisch", "Normal"]
CLASS_COLORS = [(255, 80, 80), (80, 200, 120)]  # Atypisch=red, Normal=green
META_FILENAME = ".dapp_meta.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0  # 1.0 = user-drawn

    def width(self) -> float:
        return self.x2 - self.x1

    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class ImageMeta:
    n_boxes: int = 0
    max_conf: float = 0.0
    mean_conf: float = 0.0
    edited: bool = False
    n_atypisch: int = 0
    n_normal: int = 0


def label_path_for(image_path: str | Path) -> Path:
    p = Path(image_path)
    return p.with_suffix(".txt")


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            out.append(p)
    return out


def read_yolo_labels(txt_path: str | Path, img_w: int, img_h: int) -> list[Box]:
    p = Path(txt_path)
    if not p.exists():
        return []
    boxes: list[Box] = []
    with open(p, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append(Box(cls=cls, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf))
    return boxes


def write_yolo_labels(
    txt_path: str | Path,
    boxes: Iterable[Box],
    img_w: int,
    img_h: int,
    include_conf: bool = False,
) -> None:
    p = Path(txt_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for b in boxes:
            cx = ((b.x1 + b.x2) / 2) / img_w
            cy = ((b.y1 + b.y2) / 2) / img_h
            w = (b.x2 - b.x1) / img_w
            h = (b.y2 - b.y1) / img_h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            if include_conf:
                f.write(f"{b.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {b.conf:.4f}\n")
            else:
                f.write(f"{b.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


@dataclass
class FolderMeta:
    inference_timestamp: str = ""
    model_path: str = ""
    conf_threshold: float = 0.0
    images: dict[str, ImageMeta] = field(default_factory=dict)

    @classmethod
    def load(cls, folder: str | Path) -> "FolderMeta":
        p = Path(folder) / META_FILENAME
        if not p.exists():
            return cls()
        try:
            with open(p, "r") as f:
                raw = json.load(f)
            images = {k: ImageMeta(**v) for k, v in raw.get("images", {}).items()}
            return cls(
                inference_timestamp=raw.get("inference_timestamp", ""),
                model_path=raw.get("model_path", ""),
                conf_threshold=raw.get("conf_threshold", 0.0),
                images=images,
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            return cls()

    def save(self, folder: str | Path) -> None:
        p = Path(folder) / META_FILENAME
        raw = {
            "inference_timestamp": self.inference_timestamp,
            "model_path": self.model_path,
            "conf_threshold": self.conf_threshold,
            "images": {k: asdict(v) for k, v in self.images.items()},
        }
        with open(p, "w") as f:
            json.dump(raw, f, indent=2)

    def mark_edited(self, image_name: str, n_boxes: int) -> None:
        meta = self.images.get(image_name) or ImageMeta()
        meta.edited = True
        meta.n_boxes = n_boxes
        self.images[image_name] = meta

    def clear_edit(self, image_name: str) -> None:
        meta = self.images.get(image_name)
        if meta is not None:
            meta.edited = False
