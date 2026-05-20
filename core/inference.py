from __future__ import annotations

import datetime
import os
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from .annotations import (
    Box,
    FolderMeta,
    ImageMeta,
    label_path_for,
    write_yolo_labels,
)

_DEFAULT_CHUNK = 250   # fallback when no chunk_size is passed


class InferenceWorker(QObject):
    """Runs YOLO predict() in a background QThread.

    Images are processed in chunks of chunk_size so peak memory stays bounded
    regardless of folder size.  Each chunk saves .dapp_meta.json when done,
    so a crash loses at most one chunk and the next run skips already-labeled
    images automatically.

    Emits per-image meta as inference streams. Caller is responsible for
    moving this object to a QThread and calling run() via the thread's
    started signal.
    """

    progress = Signal(int, int, str)     # current, total, image_name
    image_done = Signal(str, dict)        # image_path, meta dict
    finished = Signal(dict)               # final folder meta dict
    error = Signal(str)

    def __init__(
        self,
        model_path: str,
        image_paths: list[str],
        folder: str,
        conf: float,
        iou: float,
        batch: int,
        device: str,
        imgsz: int = 512,
        resume: bool = True,
        chunk_size: int = _DEFAULT_CHUNK,
    ):
        super().__init__()
        self.model_path = model_path
        self.image_paths = list(image_paths)
        self.folder = folder
        self.conf = conf
        self.iou = iou
        self.batch = batch
        self.device = device
        self.imgsz = imgsz
        self.resume = resume   # skip images that already have a label file
        self.chunk_size = chunk_size
        self._cancel = False

    @Slot()
    def cancel(self):
        self._cancel = True

    @Slot()
    def run(self):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            self.error.emit(f"ultralytics not installed: {e}")
            return

        if not Path(self.model_path).exists():
            self.error.emit(f"Model file not found: {self.model_path}")
            return

        try:
            model = YOLO(self.model_path)
        except Exception as e:
            self.error.emit(f"Failed to load model: {e}")
            return

        folder_meta = FolderMeta.load(self.folder)
        folder_meta.inference_timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        folder_meta.model_path = self.model_path
        folder_meta.conf_threshold = self.conf

        # When resuming, skip images that already have a label file on disk
        # (written by a previous run).  Edited images are always skipped.
        if self.resume:
            pending = [
                p for p in self.image_paths
                if not _already_labeled(p, folder_meta)
            ]
            skipped = len(self.image_paths) - len(pending)
            if skipped:
                self.progress.emit(skipped, len(self.image_paths),
                                   f"(skipped {skipped} already-labeled)")
        else:
            pending = self.image_paths

        total = len(self.image_paths)
        done = total - len(pending)   # images already handled before this run

        if not pending:
            folder_meta.save(self.folder)
            self.finished.emit(_folder_meta_dict(folder_meta))
            return

        try:
            # Process in chunks so each predict() call has a bounded memory
            # footprint — critical for large folders on memory-limited machines.
            for chunk_start in range(0, len(pending), self.chunk_size):
                if self._cancel:
                    break

                chunk = pending[chunk_start: chunk_start + self.chunk_size]
                results_iter = model.predict(
                    source=chunk,
                    conf=self.conf,
                    iou=self.iou,
                    batch=self.batch,
                    device=self.device,
                    imgsz=self.imgsz,
                    stream=True,
                    verbose=False,
                    save=False,
                )

                for i, res in enumerate(results_iter):
                    if self._cancel:
                        break

                    # Use the original path we passed in — res.path can be a
                    # temp/resolved path that differs from the user's folder.
                    img_path = chunk[i]
                    img_name = os.path.basename(img_path)
                    h, w = res.orig_shape[:2]

                    boxes: list[Box] = []
                    confs = []
                    if res.boxes is not None and len(res.boxes) > 0:
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        cls_arr = res.boxes.cls.cpu().numpy().astype(int)
                        conf_arr = res.boxes.conf.cpu().numpy()
                        for (x1, y1, x2, y2), c, cf in zip(xyxy, cls_arr, conf_arr):
                            boxes.append(Box(int(c), float(x1), float(y1),
                                            float(x2), float(y2), float(cf)))
                            confs.append(float(cf))

                    existing = folder_meta.images.get(img_name)
                    lbl_path = label_path_for(img_path)
                    if existing is None or not existing.edited:
                        write_yolo_labels(lbl_path, boxes, w, h, include_conf=True)

                    n_atypisch = sum(1 for b in boxes if b.cls == 0)
                    n_normal = sum(1 for b in boxes if b.cls == 1)
                    meta = ImageMeta(
                        n_boxes=len(boxes),
                        max_conf=max(confs) if confs else 0.0,
                        mean_conf=(sum(confs) / len(confs)) if confs else 0.0,
                        edited=existing.edited if existing else False,
                        n_atypisch=n_atypisch,
                        n_normal=n_normal,
                    )
                    folder_meta.images[img_name] = meta
                    done += 1

                    self.progress.emit(done, total, img_name)
                    self.image_done.emit(img_path, _meta_to_dict(meta))

                # Save after every chunk so a crash loses at most one chunk.
                folder_meta.save(self.folder)

            self.finished.emit(_folder_meta_dict(folder_meta))

        except Exception as e:
            # Save whatever we finished before the error.
            folder_meta.save(self.folder)
            self.error.emit(f"Inference failed: {e}")


def _already_labeled(img_path: str, folder_meta: FolderMeta) -> bool:
    """True if this image should be skipped on resume.

    Skip if:
    - The image is marked as user-edited (always preserve edits), OR
    - A non-empty .txt label file already exists on disk from a previous run.
    """
    name = os.path.basename(img_path)
    existing = folder_meta.images.get(name)
    if existing and existing.edited:
        return True
    lbl = label_path_for(img_path)
    return Path(lbl).exists()


def _meta_to_dict(m: ImageMeta) -> dict:
    return {
        "n_boxes": m.n_boxes,
        "max_conf": m.max_conf,
        "mean_conf": m.mean_conf,
        "edited": m.edited,
        "n_atypisch": m.n_atypisch,
        "n_normal": m.n_normal,
    }


def _folder_meta_dict(fm: FolderMeta) -> dict:
    return {
        "inference_timestamp": fm.inference_timestamp,
        "model_path": fm.model_path,
        "conf_threshold": fm.conf_threshold,
        "n_images": len(fm.images),
        "n_with_detections": sum(1 for m in fm.images.values() if m.n_boxes > 0),
        "n_low_conf": sum(1 for m in fm.images.values() if 0 < m.max_conf < 0.4),
        "n_edited": sum(1 for m in fm.images.values() if m.edited),
    }
