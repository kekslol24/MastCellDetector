from __future__ import annotations

import datetime
import os
import shutil
import tempfile
from pathlib import Path

import yaml
from PySide6.QtCore import QObject, Signal, Slot

from .annotations import (
    CLASS_NAMES,
    FolderMeta,
    IMAGE_EXTS,
    label_path_for,
    list_images,
)


class FineTuneWorker(QObject):
    """Fine-tune the current model on the user's edited+inferred labels.

    The naive design (fine-tune on the user's edits in isolation) is exactly
    the failure mode that produced the P2 / DL_Modell_FV problem documented
    in hpc/experiment_log.md: the model becomes very good at one new patient
    but silently regresses on prior patients (catastrophic forgetting and
    patient-specific overfitting).

    Two safeguards are implemented here:

    1. BASE-CORPUS MERGE.
       If `base_train_dir` is set, every image in that directory (with its
       label file) is symlinked into the train split alongside the user's
       edited images. This anchors fine-tuning to the original P1–P8
       distribution and prevents the model from drifting toward a single
       new patient's quirks.

    2. BASELINE VALIDATION.
       If `baseline_test_dir` is set, both the OLD model (`model_path`) and
       the NEW model (`best.pt` produced by training) are evaluated on it
       after training finishes. Per-class recall before/after is emitted
       via the `validation` signal. The UI is responsible for blocking the
       weight swap if the new model regressed beyond the user-defined
       tolerance.
    """

    progress = Signal(str)               # status message
    epoch = Signal(int, int, dict)        # current_epoch, total_epochs, metrics
    validation = Signal(dict)             # before/after recall comparison
    finished = Signal(str)               # path to new best.pt
    error = Signal(str)

    def __init__(
        self,
        model_path: str,
        folder: str,
        device: str,
        batch: int,
        epochs: int = 30,
        imgsz: int = 512,
        only_edited: bool = True,
        output_dir: str | None = None,
        base_train_dir: str | None = None,
        baseline_test_dir: str | None = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.folder = folder
        self.device = device
        self.batch = batch
        self.epochs = epochs
        self.imgsz = imgsz
        self.only_edited = only_edited
        self.output_dir = output_dir or os.path.join(folder, ".retrain_runs")
        # Path to the original training corpus (e.g. an export of P1–P8).
        # When provided, all of its labelled images are merged into the
        # train split — never into val — to anchor fine-tuning.
        self.base_train_dir = base_train_dir
        # Path to a frozen test set with verified labels. Used to measure
        # recall before/after fine-tuning so the UI can reject regressions.
        self.baseline_test_dir = baseline_test_dir
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

        try:
            train_imgs, val_imgs = self._collect_split()
        except Exception as e:
            self.error.emit(f"Could not build dataset: {e}")
            return

        if len(train_imgs) == 0:
            self.error.emit(
                "No images selected for training. "
                + ("Edit at least one image first." if self.only_edited else "Folder is empty.")
            )
            return

        # The user's edited corpus must contain at least one non-empty label
        # — otherwise we have nothing to fine-tune on. Base corpus images
        # don't count for this check (they're the anchor, not the signal).
        n_with_boxes = sum(
            1 for p in train_imgs
            if label_path_for(p).exists() and label_path_for(p).stat().st_size > 0
        )
        if n_with_boxes == 0:
            self.error.emit(
                "All label files are empty — no bounding boxes to train on.\n"
                "Re-run inference first so the .txt files are written, "
                "then open images in the editor to confirm or adjust boxes."
            )
            return

        # Optionally extend the train set with the base corpus. We keep val
        # restricted to the user's folder so val performance reflects the
        # *new* patient's data, not the old corpus (otherwise val would just
        # measure how well we're memorising the anchor).
        base_train_imgs: list[Path] = []
        if self.base_train_dir:
            base_train_imgs = self._collect_base_corpus(self.base_train_dir)
            if base_train_imgs:
                self.progress.emit(
                    f"Merging {len(base_train_imgs)} base-corpus images into train "
                    "(prevents patient-specific overfitting)."
                )
            else:
                self.progress.emit(
                    f"Warning: base corpus '{self.base_train_dir}' contained no "
                    "labelled images; fine-tuning without anchor."
                )

        self.progress.emit(
            f"Training on {len(train_imgs) + len(base_train_imgs)} images "
            f"({len(train_imgs)} edits + {len(base_train_imgs)} base), "
            f"validating on {len(val_imgs)}."
        )

        workspace = Path(tempfile.mkdtemp(prefix="dapp_ft_"))
        new_weights: str | None = None
        try:
            yaml_path = self._build_yaml(
                workspace,
                train_imgs + base_train_imgs,
                val_imgs,
            )

            run_name = f"retrain_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.progress.emit(f"Starting fine-tune ({self.epochs} epochs)...")

            model = YOLO(self.model_path)

            def _on_epoch_end(trainer):
                ep = int(getattr(trainer, "epoch", 0)) + 1
                metrics = {}
                if hasattr(trainer, "metrics") and trainer.metrics:
                    metrics = {
                        k: float(v) for k, v in trainer.metrics.items()
                        if isinstance(v, (int, float))
                    }
                self.epoch.emit(ep, self.epochs, metrics)
                if self._cancel:
                    raise RuntimeError("cancelled by user")

            model.add_callback("on_fit_epoch_end", _on_epoch_end)

            model.train(
                data=str(yaml_path),
                epochs=self.epochs,
                patience=max(5, self.epochs // 3),
                batch=self.batch,
                device=self.device,
                imgsz=self.imgsz,
                project=self.output_dir,
                name=run_name,
                exist_ok=True,
                verbose=False,
                workers=0,
                cache=False,
                augment=True,
                mosaic=0.0,
                cos_lr=True,
                lr0=0.001,
                freeze=10,
            )

            best = Path(self.output_dir) / run_name / "weights" / "best.pt"
            if not best.exists():
                self.error.emit(f"Training finished but best.pt not found at {best}")
                return
            new_weights = str(best)

            # Mandatory baseline validation. Before reporting `finished`, we
            # tell the UI how recall changed on the held-out test set so the
            # user can refuse to promote a regressed model.
            if self.baseline_test_dir:
                self.progress.emit("Validating on baseline test set...")
                comparison = self._compare_on_baseline(
                    old_weights=self.model_path,
                    new_weights=new_weights,
                    baseline_dir=self.baseline_test_dir,
                )
                if comparison is not None:
                    self.validation.emit(comparison)

            self.finished.emit(new_weights)
        except Exception as e:
            self.error.emit(f"Training failed: {e}")
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    # ----------------------------------------------------------------- splits
    def _collect_split(self) -> tuple[list[Path], list[Path]]:
        meta = FolderMeta.load(self.folder)
        all_imgs = list_images(self.folder)

        eligible: list[Path] = []
        for img in all_imgs:
            name = img.name
            lbl = label_path_for(img)
            img_meta = meta.images.get(name)
            if self.only_edited:
                if img_meta and img_meta.edited:
                    eligible.append(img)
            else:
                if lbl.exists():
                    eligible.append(img)

        if len(eligible) < 4:
            return eligible, []

        from sklearn.model_selection import train_test_split
        train, val = train_test_split(eligible, test_size=0.15, random_state=42)
        return train, val

    def _collect_base_corpus(self, base_dir: str) -> list[Path]:
        """Return all labelled images under base_dir.

        Looks for images alongside .txt label files. The directory layout is
        flat — same convention as the user's working folder. (For deeply
        nested YOLO datasets, point at the leaf images/ directory.)
        """
        root = Path(base_dir)
        if not root.exists():
            return []
        out: list[Path] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = label_path_for(p)
            # Accept images with a label file (including empty = confirmed
            # negative). Skip raw-image-only files to avoid noise.
            if lbl.exists():
                out.append(p)
        return out

    # ----------------------------------------------------------- baseline val
    def _compare_on_baseline(
        self,
        old_weights: str,
        new_weights: str,
        baseline_dir: str,
    ) -> dict | None:
        """Run val() on the baseline test set with both models, return diff.

        Returns a dict with mAP50 and per-class recall for old and new, or
        None if the baseline directory is empty / unusable.
        """
        from ultralytics import YOLO

        baseline_imgs = self._collect_base_corpus(baseline_dir)
        if not baseline_imgs:
            return None

        # Build a one-shot YOLO data config pointing at the baseline. We
        # treat its "train" and "val" splits as the same list — Ultralytics
        # requires both to be defined; only val is used for `model.val()`.
        ws = Path(tempfile.mkdtemp(prefix="dapp_baseline_"))
        try:
            img_dir = ws / "images"
            lbl_dir = ws / "labels"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            paths: list[str] = []
            for src in baseline_imgs:
                img_dst = img_dir / src.name
                if not img_dst.exists():
                    try:
                        os.symlink(src, img_dst)
                    except (OSError, NotImplementedError):
                        shutil.copy2(src, img_dst)

                src_lbl = label_path_for(src)
                lbl_dst = lbl_dir / (src.stem + ".txt")
                if not lbl_dst.exists():
                    if src_lbl.exists():
                        try:
                            os.symlink(src_lbl, lbl_dst)
                        except (OSError, NotImplementedError):
                            shutil.copy2(src_lbl, lbl_dst)
                    else:
                        lbl_dst.write_text("")
                paths.append(str(img_dst))

            list_txt = ws / "all.txt"
            list_txt.write_text("\n".join(paths))

            yaml_path = ws / "data.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(
                    {
                        "path": str(ws),
                        "train": "all.txt",
                        "val": "all.txt",
                        "nc": len(CLASS_NAMES),
                        "names": CLASS_NAMES,
                    },
                    f,
                )

            def _evaluate(weights: str) -> dict:
                m = YOLO(weights)
                res = m.val(
                    data=str(yaml_path),
                    split="val",
                    verbose=False,
                    workers=0,
                    device=self.device,
                    batch=self.batch,
                )
                box = res.box
                # box.r is per-class; pad with NaN if a class has no GT.
                rec = list(getattr(box, "r", []) or [])
                while len(rec) < len(CLASS_NAMES):
                    rec.append(float("nan"))
                return {
                    "mAP50": float(box.map50),
                    "mAP50-95": float(box.map),
                    "mean_recall": float(box.mr),
                    "recall_per_class": [float(x) for x in rec[: len(CLASS_NAMES)]],
                }

            return {
                "n_images": len(paths),
                "old": _evaluate(old_weights),
                "new": _evaluate(new_weights),
                "class_names": list(CLASS_NAMES),
            }
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    # -------------------------------------------------------------- yaml/link
    def _build_yaml(
        self,
        workspace: Path,
        train_imgs: list[Path],
        val_imgs: list[Path],
    ) -> Path:
        img_dir = workspace / "images"
        lbl_dir = workspace / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        def link(paths: list[Path]) -> list[str]:
            out = []
            for src in paths:
                base = src.name
                img_dst = img_dir / base
                # Names from the user's folder and the base corpus may collide.
                # Prefix the base-corpus link to keep them unique.
                if img_dst.exists() and not img_dst.samefile(src):
                    img_dst = img_dir / f"base_{base}"
                if not img_dst.exists():
                    try:
                        os.symlink(src, img_dst)
                    except (OSError, NotImplementedError):
                        shutil.copy2(src, img_dst)

                src_lbl = label_path_for(src)
                lbl_dst = lbl_dir / (img_dst.stem + ".txt")
                if src_lbl.exists() and not lbl_dst.exists():
                    try:
                        os.symlink(src_lbl, lbl_dst)
                    except (OSError, NotImplementedError):
                        shutil.copy2(src_lbl, lbl_dst)
                elif not src_lbl.exists() and not lbl_dst.exists():
                    # Confirmed negative: empty label file.
                    lbl_dst.write_text("")
                out.append(str(img_dst))
            return out

        train_paths = link(train_imgs)
        val_paths = link(val_imgs) if val_imgs else train_paths[: max(1, len(train_paths) // 10)]

        train_txt = workspace / "train.txt"
        val_txt = workspace / "val.txt"
        train_txt.write_text("\n".join(train_paths))
        val_txt.write_text("\n".join(val_paths))

        yaml_path = workspace / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(
                {
                    "path": str(workspace),
                    "train": "train.txt",
                    "val": "val.txt",
                    "nc": len(CLASS_NAMES),
                    "names": CLASS_NAMES,
                },
                f,
            )
        return yaml_path
