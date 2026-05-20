"""Export user corrections as a clean dataset for central re-training.

The right place for fine-tuning is *centrally*, with proper LOPO validation
across the full multi-patient corpus — not inside the desktop app on a
single new patient. The local fine-tune is a convenience for personal use
(e.g. one clinician adapting to their lab's stain). Genuine model
improvement should flow back to ZHAW for re-training.

CorrectionExporter packages the user's edited images + labels + provenance
metadata (annotator, date, source folder, model version) into a single
ZIP that can be sent to ZHAW. The metadata is what makes a future re-train
auditable: which corrections came from whom, on what model, on what date.
"""

from __future__ import annotations

import datetime
import getpass
import hashlib
import json
import platform
import zipfile
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from .annotations import FolderMeta, IMAGE_EXTS, label_path_for, list_images


def _file_hash(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-1 first 16 chars — short enough for filenames, unique enough."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()[:16]


class CorrectionExporter(QObject):
    """Bundle edited corrections into a ZIP for central submission."""

    progress = Signal(int, int, str)   # current, total, filename
    finished = Signal(str, dict)        # zip_path, manifest dict
    error = Signal(str)

    def __init__(
        self,
        folder: str,
        zip_path: str,
        annotator: str = "",
        notes: str = "",
        model_path: str = "",
        source_label: str = "",
    ):
        super().__init__()
        self.folder = Path(folder)
        self.zip_path = Path(zip_path)
        self.annotator = annotator or getpass.getuser()
        self.notes = notes
        self.model_path = model_path
        # Free-text label identifying the source (e.g. "USZ patient 14",
        # "Lab Hagenholz batch 2026-04"). Optional but recommended.
        self.source_label = source_label

    @Slot()
    def run(self):
        try:
            meta = FolderMeta.load(str(self.folder))
            all_imgs = list_images(self.folder)

            # Only export images the user actually touched. Inference-only
            # outputs are not corrections and have no signal value for
            # central re-training.
            edited = []
            for img in all_imgs:
                im = meta.images.get(img.name)
                if im and im.edited:
                    edited.append(img)

            if not edited:
                self.error.emit(
                    "No edited images to export. Use the annotation editor "
                    "to confirm or correct boxes first; only edited images "
                    "are included so we don't ship unverified inference."
                )
                return

            self.zip_path.parent.mkdir(parents=True, exist_ok=True)

            manifest = self._build_manifest(edited, meta)

            with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Manifest first so reviewers see provenance up-front.
                zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
                zf.writestr("README.txt", _README)

                total = len(edited)
                for i, img in enumerate(edited, start=1):
                    arc_img = f"images/{img.name}"
                    zf.write(img, arc_img)

                    lbl = label_path_for(img)
                    arc_lbl = f"labels/{img.stem}.txt"
                    if lbl.exists():
                        zf.write(lbl, arc_lbl)
                    else:
                        # Edited-to-empty = confirmed negative. Preserve.
                        zf.writestr(arc_lbl, "")

                    self.progress.emit(i, total, img.name)

            self.finished.emit(str(self.zip_path), manifest)
        except Exception as e:
            self.error.emit(f"Export failed: {e}")

    def _build_manifest(self, edited: list[Path], meta: FolderMeta) -> dict:
        n_with_boxes = sum(
            1 for p in edited
            if label_path_for(p).exists() and label_path_for(p).stat().st_size > 0
        )
        n_empty = len(edited) - n_with_boxes

        model_info = {"path": self.model_path}
        if self.model_path and Path(self.model_path).exists():
            model_info["sha1_16"] = _file_hash(Path(self.model_path))
            model_info["size_bytes"] = Path(self.model_path).stat().st_size

        return {
            "schema": "dapp.corrections/1",
            "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "annotator": self.annotator,
            "host": platform.node(),
            "source_label": self.source_label,
            "notes": self.notes,
            "source_folder": str(self.folder),
            "model": model_info,
            "inference_run": {
                "timestamp": meta.inference_timestamp,
                "model_path": meta.model_path,
                "conf_threshold": meta.conf_threshold,
            },
            "counts": {
                "n_edited": len(edited),
                "n_with_boxes": n_with_boxes,
                "n_confirmed_empty": n_empty,
            },
            "images": [
                {
                    "filename": img.name,
                    "n_boxes": (meta.images.get(img.name).n_boxes
                                if meta.images.get(img.name) else 0),
                }
                for img in edited
            ],
        }


_README = """\
Mast Cell Detector — corrections export
=======================================

This archive contains user-corrected detections from the desktop app.
Use it to centrally re-train a new model version with proper LOPO
cross-validation against the full P1–P8 corpus.

DO NOT promote a model fine-tuned only on this archive without
validating it on a held-out test set. See hpc/experiment_log.md
("P1–P8 Full Dataset — Class Imbalance Strategy", section
'Negative-set provenance') and the thesis 'Continual Learning Without
Distribution Drift' subsection for why.

Layout:
  MANIFEST.json   — provenance: annotator, date, source folder, model hash
  images/         — original tiles edited by the annotator
  labels/         — YOLO-format labels (empty .txt = confirmed negative)
"""
