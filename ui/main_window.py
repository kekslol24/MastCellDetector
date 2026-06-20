from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSettings, QThread, Signal, Slot
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..core.annotations import (
    FolderMeta,
    ImageMeta,
    label_path_for,
    list_images,
)
from ..core.drive import DriveDownloadWorker
from ..core.hardware import detect_hardware, format_hardware
from ..core.inference import InferenceWorker

from .annotation_editor import AnnotationEditor
from .statistics_view import StatisticsView
from .widgets import GalleryDelegate, GalleryModel, GalleryView, ThumbnailCache


FILTERS = [
    ("all", "All"),
    ("with", "With detections"),
    ("none", "No detection"),
    ("low", "Low confidence"),
    ("atypisch", "Atypisch"),
    ("normal", "Normal"),
    ("edited", "Edited"),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mast Cell Detector — Desktop")
        self.resize(1500, 900)

        self._hw = detect_hardware()
        self._folder: Optional[Path] = None
        self._model_path: str = self._guess_model_path()
        self._meta = FolderMeta()
        self._all_items: list[tuple[Path, ImageMeta]] = []
        self._active_filter: str = "all"

        self._inference_thread: Optional[QThread] = None
        self._inference_worker: Optional[InferenceWorker] = None
        self._drive_thread: Optional[QThread] = None
        self._drive_worker: Optional[DriveDownloadWorker] = None
        self._editor_source: str = "gallery"  # "gallery" or "stats"
        self._fp_undo_stack: list[tuple[str, str, ImageMeta]] = []

        self._build_ui()
        self._refresh_hardware_label()
        self._refresh_model_label()
        self._load_settings()
        self._update_action_states()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = self._build_sidebar()
        root.addWidget(sidebar)

        # Right: stacked widget with gallery & editor.
        right = QWidget()
        rlayout = QVBoxLayout(right)
        rlayout.setContentsMargins(0, 0, 0, 0)
        rlayout.setSpacing(0)

        # Filter bar
        self.filter_bar = QFrame()
        self.filter_bar.setStyleSheet("background-color: #15171c; border-bottom: 1px solid #2a2d36;")
        fb = QHBoxLayout(self.filter_bar)
        fb.setContentsMargins(10, 6, 10, 6)
        fb.setSpacing(6)
        self.filter_group = QButtonGroup(self)
        self.filter_group.setExclusive(True)
        for key, label in FILTERS:
            btn = QPushButton(label)
            btn.setObjectName("FilterChip")
            btn.setCheckable(True)
            if key == "all":
                btn.setChecked(True)
            btn.clicked.connect(lambda _ck=False, k=key: self._on_filter(k))
            fb.addWidget(btn)
            self.filter_group.addButton(btn)
        fb.addStretch()
        self.undo_fp_btn = QPushButton("↩ Undo FP")
        self.undo_fp_btn.setObjectName("FilterChip")
        self.undo_fp_btn.setToolTip("Undo last False Positive deletion")
        self.undo_fp_btn.setVisible(False)
        self.undo_fp_btn.clicked.connect(self._on_undo_fp)
        fb.addWidget(self.undo_fp_btn)

        self.summary_label = QLabel("No folder loaded")
        self.summary_label.setStyleSheet("color: #b8bcc8;")
        fb.addWidget(self.summary_label)

        # Statistics toggle — separate from the FILTERS group because it
        # switches *view*, not which images are filtered.
        self.stats_btn = QPushButton("Statistics")
        self.stats_btn.setObjectName("FilterChip")
        self.stats_btn.setCheckable(True)
        self.stats_btn.setToolTip(
            "Patient summary: Atypisch fraction vs WHO 25% threshold,\n"
            "per-image cell counts, confidence distribution."
        )
        self.stats_btn.clicked.connect(self._on_toggle_stats)
        fb.addWidget(self.stats_btn)

        rlayout.addWidget(self.filter_bar)

        # Stack: gallery + editor + statistics
        self.stack = QStackedWidget()
        self.thumb_cache = ThumbnailCache(max_workers=4, parent=self)
        self.gallery_model = GalleryModel(self.thumb_cache, self)
        self.gallery = GalleryView()
        self.gallery.setModel(self.gallery_model)
        self.gallery.setItemDelegate(GalleryDelegate(self.gallery))
        self.gallery.image_activated.connect(self._open_editor_for)
        self.gallery.context_menu_requested.connect(self._on_gallery_context_menu)
        self.gallery.delete_requested.connect(self._mark_as_fp)

        self.editor = AnnotationEditor()
        self.editor.back_requested.connect(self._on_editor_back)
        self.editor.saved.connect(self._on_image_saved)

        self.stats_view = StatisticsView()
        self.stats_view.open_image_requested.connect(self._open_editor_from_stats)

        self.stack.addWidget(self.gallery)
        self.stack.addWidget(self.editor)
        self.stack.addWidget(self.stats_view)
        rlayout.addWidget(self.stack, 1)

        root.addWidget(right, 1)

        self.setCentralWidget(central)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(280)
        self.progress.setVisible(False)
        self.status.addPermanentWidget(self.progress)

    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("Sidebar")
        side.setFixedWidth(310)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title = QLabel("Mast Cell Detector")
        title.setObjectName("SidebarTitle")
        layout.addWidget(title)

        # SOURCE
        layout.addWidget(self._section("Source"))
        srow = QHBoxLayout()
        srow.setContentsMargins(14, 0, 14, 4)
        self.open_folder_btn = QPushButton("📁  Open Folder")
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        srow.addWidget(self.open_folder_btn)
        layout.addLayout(srow)

        drow = QHBoxLayout()
        drow.setContentsMargins(14, 0, 14, 4)
        self.drive_input = QLineEdit()
        self.drive_input.setPlaceholderText("Paste Google Drive link…")
        drow.addWidget(self.drive_input)
        self.drive_btn = QPushButton("⇣")
        self.drive_btn.setFixedWidth(34)
        self.drive_btn.clicked.connect(self._on_drive_download)
        drow.addWidget(self.drive_btn)
        layout.addLayout(drow)

        self.source_label = QLabel("(no source loaded)")
        self.source_label.setObjectName("InfoLabel")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        # HARDWARE
        layout.addWidget(self._section("Hardware"))
        self.hw_label = QLabel("")
        self.hw_label.setObjectName("InfoLabel")
        self.hw_label.setWordWrap(True)
        layout.addWidget(self.hw_label)

        # MODEL
        layout.addWidget(self._section("Model"))
        self.model_label = QLabel("")
        self.model_label.setObjectName("InfoLabel")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        mrow = QHBoxLayout()
        mrow.setContentsMargins(14, 0, 14, 4)
        self.model_btn = QPushButton("Browse…")
        self.model_btn.clicked.connect(self._on_select_model)
        mrow.addWidget(self.model_btn)
        layout.addLayout(mrow)

        # INFERENCE PARAMS
        layout.addWidget(self._section("Inference"))
        prow = QHBoxLayout()
        prow.setContentsMargins(14, 0, 14, 4)
        prow.addWidget(QLabel("Confidence"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        prow.addWidget(self.conf_spin)
        layout.addLayout(prow)

        irow = QHBoxLayout()
        irow.setContentsMargins(14, 0, 14, 4)
        irow.addWidget(QLabel("IoU NMS"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setToolTip(
            "IoU threshold for non-maximum suppression.\n"
            "When two boxes overlap more than this, the one with lower\n"
            "confidence is removed. Lower = fewer overlapping boxes."
        )
        irow.addWidget(self.iou_spin)
        layout.addLayout(irow)

        brow = QHBoxLayout()
        brow.setContentsMargins(14, 0, 14, 4)
        brow.addWidget(QLabel("Batch"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(self._hw["recommended_batch"])
        brow.addWidget(self.batch_spin)
        layout.addLayout(brow)

        crow = QHBoxLayout()
        crow.setContentsMargins(14, 0, 14, 4)
        crow.addWidget(QLabel("Chunk size"))
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(10, 2000)
        self.chunk_spin.setValue(250)
        self.chunk_spin.setToolTip(
            "Images per YOLO predict() call.\n"
            "Lower = less peak RAM, higher = faster throughput.\n"
            "250 is safe for most systems with ≥8 GB RAM."
        )
        crow.addWidget(self.chunk_spin)
        layout.addLayout(crow)

        runrow = QHBoxLayout()
        runrow.setContentsMargins(14, 6, 14, 4)
        self.run_btn = QPushButton("▶  Run Inference")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self._on_run_inference)
        runrow.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.setObjectName("DangerButton")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_inference)
        runrow.addWidget(self.cancel_btn)
        layout.addLayout(runrow)

        layout.addStretch()
        return side

    @staticmethod
    def _section(label: str) -> QLabel:
        l = QLabel(label)
        l.setObjectName("SidebarSection")
        return l

    # ---------------------------------------------------------- helpers
    def _guess_model_path(self) -> str:
        base = Path(__file__).parent.parent.parent   # BA/
        candidates = [
            base / "hpc" / "DL_Modell_FV.pt",
            base / "app" / "DL_Modell_FV.pt",
            base / "DL_Modell_FV.pt",
            base.parent / "DL_Modell_FV.pt",        # one level above BA/
        ]
        for c in candidates:
            if c.exists():
                return str(c.resolve())
        return ""   # no fallback — force the user to pick via Browse

    def _refresh_model_label(self):
        if self._model_path and Path(self._model_path).exists():
            name = Path(self._model_path).name
            size_mb = Path(self._model_path).stat().st_size / (1024 ** 2)
            self.model_label.setText(
                f"{name}  ({size_mb:.1f} MB)\n{Path(self._model_path).parent}"
            )
        elif self._model_path:
            self.model_label.setText(f"⚠ not found: {self._model_path}\nUse Browse… to select a .pt file")
        else:
            self.model_label.setText("⚠ No model selected — use Browse… to pick a .pt file")

    def _refresh_hardware_label(self):
        self.hw_label.setText(format_hardware(self._hw))

    def _update_action_states(self):
        has_folder = self._folder is not None
        running_inf = self._inference_thread is not None
        running_drive = self._drive_thread is not None
        any_running = running_inf or running_drive

        self.run_btn.setEnabled(has_folder and not any_running)
        self.cancel_btn.setEnabled(running_inf)
        self.open_folder_btn.setEnabled(not any_running)
        self.drive_btn.setEnabled(not any_running)
        self.model_btn.setEnabled(not any_running)

    # ---------------------------------------------------------- folder loading
    def _on_open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose image folder", str(Path.home()))
        if path:
            self._load_folder(path)

    def _load_folder(self, path: str):
        self._folder = Path(path)
        self.editor.open_folder(path)
        self._meta = FolderMeta.load(self._folder)
        self._fp_undo_stack.clear()
        self.undo_fp_btn.setVisible(False)

        imgs = list_images(self._folder)
        self._all_items = [(p, self._meta.images.get(p.name, ImageMeta())) for p in imgs]
        self._apply_filter()

        n_imgs = len(imgs)
        n_with = sum(1 for _, m in self._all_items if m.n_boxes > 0)
        n_edit = sum(1 for _, m in self._all_items if m.edited)
        self.source_label.setText(
            f"{self._folder}\n{n_imgs} images · {n_with} with detections · {n_edit} edited"
        )
        self._update_summary()
        self._update_action_states()
        # Initial stats pass — picks up any pre-existing labels even
        # before inference is re-run.
        self.stats_view.refresh(self._folder)
        self.status.showMessage(f"Loaded {n_imgs} images from {self._folder.name}", 5000)

    def _update_summary(self):
        n = len(self._all_items)
        nw = sum(1 for _, m in self._all_items if m.n_boxes > 0)
        nl = sum(1 for _, m in self._all_items if 0 < m.max_conf < 0.4)
        ne = sum(1 for _, m in self._all_items if m.edited)
        self.summary_label.setText(
            f"{n} total · {nw} detect · {nl} low-conf · {ne} edited"
        )

    # ---------------------------------------------------------- filters
    def _on_filter(self, key: str):
        self._active_filter = key
        self._apply_filter()

    def _apply_filter(self):
        items = self._all_items
        k = self._active_filter
        if k == "with":
            items = [it for it in items if it[1].n_boxes > 0]
        elif k == "none":
            items = [it for it in items if it[1].n_boxes == 0]
        elif k == "low":
            items = [it for it in items if 0 < it[1].max_conf < 0.4]
        elif k == "atypisch":
            items = [it for it in items if it[1].n_atypisch > 0]
        elif k == "normal":
            items = [it for it in items if it[1].n_normal > 0]
        elif k == "edited":
            items = [it for it in items if it[1].edited]
        self.gallery_model.set_items(items)

    # ---------------------------------------------------------- model
    def _on_select_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose YOLO weights", str(Path.home()), "YOLO weights (*.pt);;All files (*)"
        )
        if path:
            self._model_path = path
            self._refresh_model_label()

    # ---------------------------------------------------------- drive
    def _on_drive_download(self):
        link = self.drive_input.text().strip()
        if not link:
            QMessageBox.information(self, "Drive", "Paste a Google Drive folder link first.")
            return

        dest = QFileDialog.getExistingDirectory(
            self, "Where to save downloaded folder?", str(Path.home())
        )
        if not dest:
            return

        self._drive_thread = QThread(self)
        self._drive_worker = DriveDownloadWorker(link, dest)
        self._drive_worker.moveToThread(self._drive_thread)
        self._drive_thread.started.connect(self._drive_worker.run)
        self._drive_worker.progress.connect(lambda msg: self.status.showMessage(msg))
        self._drive_worker.finished.connect(self._on_drive_done)
        self._drive_worker.error.connect(self._on_drive_error)
        self._drive_worker.finished.connect(self._drive_thread.quit)
        self._drive_worker.error.connect(self._drive_thread.quit)
        self._drive_thread.finished.connect(self._cleanup_drive)

        self.status.showMessage("Downloading from Drive…")
        self._update_action_states()
        self._drive_thread.start()

    def _on_drive_done(self, local_path: str):
        self.status.showMessage(f"Downloaded to {local_path}", 5000)
        self._load_folder(local_path)

    def _on_drive_error(self, msg: str):
        QMessageBox.warning(self, "Drive download failed", msg)

    def _cleanup_drive(self):
        if self._drive_worker:
            self._drive_worker.deleteLater()
        if self._drive_thread:
            self._drive_thread.deleteLater()
        self._drive_worker = None
        self._drive_thread = None
        self._update_action_states()

    # ---------------------------------------------------------- inference
    def _on_run_inference(self):
        if self._folder is None:
            return
        if not self._model_path or not Path(self._model_path).exists():
            QMessageBox.warning(
                self, "No model selected",
                "No valid model weights found.\nUse Browse… in the Model section to select your DL_Modell_FV.pt file."
            )
            return

        edited = sum(1 for _, m in self._all_items if m.edited)
        if edited > 0:
            ans = QMessageBox.question(
                self, "Re-run inference?",
                f"{edited} image(s) have user-edited labels. Edited labels will be PRESERVED. "
                "Re-run inference on the rest?",
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        image_paths = [str(p) for p, _ in self._all_items]
        self.progress.setMaximum(len(image_paths))
        self.progress.setValue(0)
        self.progress.setVisible(True)

        self._inference_thread = QThread(self)
        self._inference_worker = InferenceWorker(
            model_path=self._model_path,
            image_paths=image_paths,
            folder=str(self._folder),
            conf=float(self.conf_spin.value()),
            iou=float(self.iou_spin.value()),
            batch=int(self.batch_spin.value()),
            device=self._hw["device"],
            imgsz=512,
            chunk_size=int(self.chunk_spin.value()),
        )
        self._inference_worker.moveToThread(self._inference_thread)
        self._inference_thread.started.connect(self._inference_worker.run)
        self._inference_worker.progress.connect(self._on_infer_progress)
        self._inference_worker.image_done.connect(self._on_image_done)
        self._inference_worker.finished.connect(self._on_infer_finished)
        self._inference_worker.error.connect(self._on_infer_error)
        self._inference_worker.finished.connect(self._inference_thread.quit)
        self._inference_worker.error.connect(self._inference_thread.quit)
        self._inference_thread.finished.connect(self._cleanup_inference)

        self._update_action_states()
        self.status.showMessage("Running inference…")
        self._inference_thread.start()

    def _on_cancel_inference(self):
        if self._inference_worker:
            self._inference_worker.cancel()
            self.status.showMessage("Cancelling…")

    def _on_infer_progress(self, current: int, total: int, name: str):
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        self.status.showMessage(f"[{current}/{total}] {name}")

    def _on_image_done(self, path: str, meta_dict: dict):
        # Match by filename — YOLO's res.path may differ from str(p) due to
        # path normalisation differences on WSL/Windows mounts.
        meta = ImageMeta(**meta_dict)
        name = Path(path).name
        for i, (p, _) in enumerate(self._all_items):
            if p.name == name:
                self._all_items[i] = (p, meta)
                self.gallery_model.update_meta(str(p), meta)
                return

    def _on_infer_finished(self, summary: dict):
        self.progress.setVisible(False)
        if summary:
            n = summary.get("n_images", 0)
            nw = summary.get("n_with_detections", 0)
            nl = summary.get("n_low_conf", 0)
            ne = summary.get("n_edited", 0)
            self.status.showMessage(
                f"Done. {n} images · {nw} with detections · {nl} low-conf · {ne} edited",
                10000,
            )
        # Reload meta from disk (worker saved it) — source of truth.
        if self._folder:
            self._meta = FolderMeta.load(self._folder)
            imgs = [p for p, _ in self._all_items]
            self._all_items = [
                (p, self._meta.images.get(p.name, ImageMeta()))
                for p in imgs
            ]
            # Force thumbnails to re-render with boxes.
            self.thumb_cache.clear()
            # Patient summary is the headline output — refresh now so the
            # user can switch to the Statistics tab and see the verdict.
            self.stats_view.refresh(self._folder)
        self._update_summary()
        self._apply_filter()

    def _on_infer_error(self, msg: str):
        self.progress.setVisible(False)
        QMessageBox.warning(self, "Inference error", msg)

    def _cleanup_inference(self):
        if self._inference_worker:
            self._inference_worker.deleteLater()
        if self._inference_thread:
            self._inference_thread.deleteLater()
        self._inference_worker = None
        self._inference_thread = None
        self._update_action_states()

    # ---------------------------------------------------------- gallery context menu
    def _on_gallery_context_menu(self, path: str):
        name = Path(path).name
        meta = self._meta.images.get(name)

        menu = QMenu(self)
        if meta and meta.edited:
            clear_action = menu.addAction("Clear edit status")
            clear_action.triggered.connect(lambda: self._clear_edit(path))
            menu.addSeparator()
        if meta and meta.n_boxes > 0:
            fp_action = menu.addAction("⚡ Mark as False Positive (delete all boxes)")
            fp_action.triggered.connect(lambda: self._mark_as_fp(path))
            menu.addSeparator()
        delete_action = menu.addAction("Delete image")
        delete_action.triggered.connect(lambda: self._delete_image(path))
        menu.exec(QCursor.pos())

    def _clear_edit(self, path: str):
        name = Path(path).name
        self._meta.clear_edit(name)
        if self._folder:
            self._meta.save(self._folder)
        for i, (p, _) in enumerate(self._all_items):
            if p.name == name:
                self._all_items[i] = (p, self._meta.images.get(name, ImageMeta()))
                break
        self.thumb_cache.invalidate(path)
        self._update_summary()
        self._apply_filter()
        self.status.showMessage(f"Edit cleared for {name}", 3000)

    def _delete_image(self, path: str):
        name = Path(path).name
        ans = QMessageBox.question(
            self, "Delete image?",
            f"Permanently delete {name} and its label file from disk?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        p = Path(path)
        txt = label_path_for(path)
        try:
            p.unlink(missing_ok=True)
            Path(txt).unlink(missing_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "Delete failed", str(e))
            return
        if name in self._meta.images:
            del self._meta.images[name]
        if self._folder:
            self._meta.save(self._folder)
        self._all_items = [(p2, m) for p2, m in self._all_items if p2.name != name]
        self.thumb_cache.invalidate(path)
        self._update_summary()
        self._apply_filter()
        self.status.showMessage(f"Deleted {name}", 4000)

    def _mark_as_fp(self, path: str):
        name = Path(path).name
        txt_path = label_path_for(path)

        original_content = ""
        try:
            if Path(txt_path).exists():
                original_content = Path(txt_path).read_text()
        except OSError:
            pass

        orig = self._meta.images.get(name)
        original_meta = ImageMeta(
            n_boxes=orig.n_boxes if orig else 0,
            max_conf=orig.max_conf if orig else 0.0,
            mean_conf=orig.mean_conf if orig else 0.0,
            edited=orig.edited if orig else False,
            n_atypisch=orig.n_atypisch if orig else 0,
            n_normal=orig.n_normal if orig else 0,
        )

        try:
            Path(txt_path).write_text("")
        except OSError as e:
            QMessageBox.warning(self, "Mark FP failed", str(e))
            return

        new_meta = ImageMeta(n_boxes=0, edited=True)
        self._meta.images[name] = new_meta
        if self._folder:
            self._meta.save(self._folder)

        for i, (p, _) in enumerate(self._all_items):
            if p.name == name:
                self._all_items[i] = (p, new_meta)
                self.thumb_cache.invalidate(str(p))
                self.gallery_model.update_meta(str(p), new_meta)
                break

        self._fp_undo_stack.append((path, original_content, original_meta))
        self.undo_fp_btn.setVisible(True)
        self._update_summary()
        self._apply_filter()
        if self._folder:
            self.stats_view.refresh(self._folder)
        self.status.showMessage(f"Marked as FP: {name}", 3000)

    def _on_undo_fp(self):
        if not self._fp_undo_stack:
            self.undo_fp_btn.setVisible(False)
            return

        path, original_content, original_meta = self._fp_undo_stack.pop()
        name = Path(path).name
        txt_path = label_path_for(path)

        try:
            Path(txt_path).write_text(original_content)
        except OSError as e:
            QMessageBox.warning(self, "Undo failed", str(e))
            self._fp_undo_stack.append((path, original_content, original_meta))
            return

        self._meta.images[name] = original_meta
        if self._folder:
            self._meta.save(self._folder)

        for i, (p, _) in enumerate(self._all_items):
            if p.name == name:
                self._all_items[i] = (p, original_meta)
                self.thumb_cache.invalidate(str(p))
                self.gallery_model.update_meta(str(p), original_meta)
                break

        if not self._fp_undo_stack:
            self.undo_fp_btn.setVisible(False)

        self._update_summary()
        self._apply_filter()
        if self._folder:
            self.stats_view.refresh(self._folder)
        self.status.showMessage(f"Undone FP: {name} restored", 3000)

    # ---------------------------------------------------------- editor
    def _open_editor_for(self, path: str):
        self.editor.set_image_list(self.gallery_model.image_paths())
        self.editor.open_image(path)
        self.stack.setCurrentWidget(self.editor)
        self.filter_bar.setVisible(False)
        self._editor_source = "gallery"
        self.stats_btn.setChecked(False)

    def _open_editor_from_stats(self, path: str):
        """Open editor from the stats table; navigates through table's sorted order."""
        if self._folder:
            image_list = self.stats_view.current_image_paths(self._folder)
            self.editor.set_image_list(image_list if image_list else self.gallery_model.image_paths())
        else:
            self.editor.set_image_list(self.gallery_model.image_paths())
        self.editor.open_image(path)
        self.stack.setCurrentWidget(self.editor)
        self.filter_bar.setVisible(False)
        self._editor_source = "stats"
        self.stats_btn.setChecked(False)

    def _on_editor_back(self):
        if self._editor_source == "stats":
            self._show_stats()
        else:
            self._show_gallery()

    def _show_stats(self):
        self.stack.setCurrentWidget(self.stats_view)
        self.filter_bar.setVisible(True)
        for btn in self.filter_group.buttons():
            btn.setVisible(False)
        self.stats_btn.setChecked(True)
        self._editor_source = "gallery"

    def _show_gallery(self):
        self.stack.setCurrentWidget(self.gallery)
        self.filter_bar.setVisible(True)
        # Restore the filter chips that the stats view hides.
        for btn in self.filter_group.buttons():
            btn.setVisible(True)
        self.stats_btn.setChecked(False)
        # Refresh in case meta changed
        self._meta = FolderMeta.load(self._folder) if self._folder else FolderMeta()
        for i, (p, _) in enumerate(self._all_items):
            new_meta = self._meta.images.get(p.name) or self._all_items[i][1]
            self._all_items[i] = (p, new_meta)
        self._update_summary()
        self._apply_filter()

    def _on_image_saved(self, path: str, n_boxes: int):
        # Reflect edit status into gallery model immediately.
        meta = self._meta.images.get(Path(path).name) if self._folder else None
        if meta is None:
            meta = ImageMeta(n_boxes=n_boxes, edited=True)
        self.gallery_model.update_meta(path, meta)
        self.status.showMessage(f"Saved {Path(path).name} ({n_boxes} boxes)", 3000)
        # Patient summary depends on edited labels — refresh whenever
        # the user changes any box. Cheap (filesystem-only walk).
        if self._folder:
            self.stats_view.refresh(self._folder)

    # ---------------------------------------------------------- statistics tab
    def _on_toggle_stats(self):
        if self.stats_btn.isChecked():
            self.stack.setCurrentWidget(self.stats_view)
            for btn in self.filter_group.buttons():
                btn.setVisible(False)
        else:
            self.stack.setCurrentWidget(self.gallery)
            for btn in self.filter_group.buttons():
                btn.setVisible(True)

    # ---------------------------------------------------------- settings persistence
    def _load_settings(self):
        s = QSettings("ZHAW", "MastCellDetector")
        self.conf_spin.setValue(float(s.value("inference/conf", self.conf_spin.value())))
        self.iou_spin.setValue(float(s.value("inference/iou", self.iou_spin.value())))
        self.batch_spin.setValue(int(s.value("inference/batch", self.batch_spin.value())))
        self.chunk_spin.setValue(int(s.value("inference/chunk_size", self.chunk_spin.value())))
        saved_model = s.value("model/path", "")
        if saved_model and Path(saved_model).exists():
            self._model_path = saved_model
            self._refresh_model_label()

    def _save_settings(self):
        s = QSettings("ZHAW", "MastCellDetector")
        s.setValue("inference/conf", self.conf_spin.value())
        s.setValue("inference/iou", self.iou_spin.value())
        s.setValue("inference/batch", self.batch_spin.value())
        s.setValue("inference/chunk_size", self.chunk_spin.value())
        if self._model_path:
            s.setValue("model/path", self._model_path)

    # ---------------------------------------------------------- shutdown
    def closeEvent(self, event):
        self._save_settings()
        for w in (self._inference_worker, self._drive_worker):
            if w and hasattr(w, "cancel"):
                try:
                    w.cancel()
                except Exception:
                    pass
        super().closeEvent(event)
