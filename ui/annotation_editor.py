from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImageReader,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.annotations import (
    CLASS_COLORS,
    CLASS_NAMES,
    Box,
    FolderMeta,
    label_path_for,
    list_images,
    read_yolo_labels,
    write_yolo_labels,
)


HANDLE_SIZE = 10


class BoxItem(QGraphicsRectItem):
    """Movable, selectable bbox with corner resize handles."""

    def __init__(self, rect: QRectF, cls: int, conf: float = 1.0):
        super().__init__(rect)
        self.cls = cls
        self.conf = conf
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self._resizing = False
        self._resize_corner = None
        self._press_pos = QPointF()
        self._orig_rect = QRectF()
        self._update_pen()

    def _update_pen(self):
        col = CLASS_COLORS[self.cls % len(CLASS_COLORS)]
        color = QColor(*col)
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(40)
        self.setBrush(QBrush(fill))

    def set_class(self, cls: int):
        self.cls = cls
        self._update_pen()
        self.update()

    def absolute_rect(self) -> QRectF:
        # Account for moves: combine pos + rect.
        r = self.rect()
        p = self.pos()
        return QRectF(r.x() + p.x(), r.y() + p.y(), r.width(), r.height())

    def commit_position(self):
        """Bake pos() into rect() so we always work in scene coords cleanly."""
        if self.pos() != QPointF(0, 0):
            r = self.rect()
            p = self.pos()
            self.setRect(QRectF(r.x() + p.x(), r.y() + p.y(), r.width(), r.height()))
            self.setPos(0, 0)

    # --- corner handle painting ---
    def paint(self, painter: QPainter, option, widget=None):
        super().paint(painter, option, widget)
        if not self.isSelected():
            return
        r = self.rect()
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.setPen(QPen(QColor("#000000"), 1))
        s = HANDLE_SIZE / max(self.scene().views()[0].transform().m11(), 0.01) if self.scene() and self.scene().views() else HANDLE_SIZE
        for cx, cy in [(r.left(), r.top()), (r.right(), r.top()),
                       (r.left(), r.bottom()), (r.right(), r.bottom())]:
            painter.drawRect(QRectF(cx - s / 2, cy - s / 2, s, s))

    def _hit_corner(self, pos: QPointF) -> Optional[str]:
        r = self.rect()
        s = HANDLE_SIZE / max(self.scene().views()[0].transform().m11(), 0.01) if self.scene() and self.scene().views() else HANDLE_SIZE
        corners = {
            "tl": (r.left(), r.top()),
            "tr": (r.right(), r.top()),
            "bl": (r.left(), r.bottom()),
            "br": (r.right(), r.bottom()),
        }
        for name, (cx, cy) in corners.items():
            if abs(pos.x() - cx) <= s and abs(pos.y() - cy) <= s:
                return name
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isSelected():
            corner = self._hit_corner(event.pos())
            if corner:
                self._resizing = True
                self._resize_corner = corner
                self._press_pos = event.pos()
                self._orig_rect = QRectF(self.rect())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            r = QRectF(self._orig_rect)
            p = event.pos()
            if self._resize_corner == "tl":
                r.setTopLeft(p)
            elif self._resize_corner == "tr":
                r.setTopRight(p)
            elif self._resize_corner == "bl":
                r.setBottomLeft(p)
            elif self._resize_corner == "br":
                r.setBottomRight(p)
            self.setRect(r.normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            self._resize_corner = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class EditorScene(QGraphicsScene):
    """Scene with click-and-drag to draw a new box when in 'add' mode."""

    box_added = Signal(BoxItem)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._add_mode = False
        self._add_class = 0
        self._draft: Optional[QGraphicsRectItem] = None
        self._draft_origin = QPointF()
        self._image_rect = QRectF()

    def set_add_mode(self, on: bool, cls: int = 0):
        self._add_mode = on
        self._add_class = cls
        for v in self.views():
            v.viewport().setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def set_image_rect(self, rect: QRectF):
        self._image_rect = rect

    def mousePressEvent(self, event):
        if self._add_mode and event.button() == Qt.MouseButton.LeftButton:
            self._draft_origin = event.scenePos()
            self._draft = QGraphicsRectItem(QRectF(self._draft_origin, self._draft_origin))
            col = CLASS_COLORS[self._add_class % len(CLASS_COLORS)]
            pen = QPen(QColor(*col), 2)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            self._draft.setPen(pen)
            self.addItem(self._draft)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._draft is not None:
            r = QRectF(self._draft_origin, event.scenePos()).normalized()
            self._draft.setRect(r)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._draft is not None and event.button() == Qt.MouseButton.LeftButton:
            r = self._draft.rect()
            self.removeItem(self._draft)
            self._draft = None
            if r.width() >= 4 and r.height() >= 4:
                # Clip to image bounds.
                if not self._image_rect.isNull():
                    r = r.intersected(self._image_rect)
                if r.width() >= 4 and r.height() >= 4:
                    box = BoxItem(r, self._add_class)
                    self.addItem(box)
                    self.box_added.emit(box)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ZoomGraphicsView(QGraphicsView):
    """View with mouse-wheel zoom and middle-button pan."""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#0e1014")))

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event):
        # Hold Space to pan with the mouse (drop into ScrollHandDrag).
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().keyReleaseEvent(event)


class AnnotationEditor(QWidget):
    """Editor view: image with editable bboxes + toolbar."""

    back_requested = Signal()
    saved = Signal(str, int)  # image_path, n_boxes (after save)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder: Optional[Path] = None
        self._images: list[Path] = []
        self._index: int = 0
        self._current_path: Optional[Path] = None
        self._img_w: int = 0
        self._img_h: int = 0
        self._dirty: bool = False

        self._build_ui()
        self._wire_shortcuts()

    def _build_ui(self):
        toolbar = QFrame()
        toolbar.setObjectName("EditorToolbar")
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(8, 4, 8, 4)
        tl.setSpacing(6)

        self.back_btn = QPushButton("← Back to gallery")
        self.back_btn.clicked.connect(self._on_back)
        tl.addWidget(self.back_btn)

        tl.addSpacing(20)

        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self.prev_image)
        tl.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_image)
        tl.addWidget(self.next_btn)

        tl.addSpacing(20)

        self.add_btn = QPushButton("✚ Add Box")
        self.add_btn.setCheckable(True)
        self.add_btn.toggled.connect(self._on_add_toggled)
        tl.addWidget(self.add_btn)

        self.class_combo = QComboBox()
        for i, name in enumerate(CLASS_NAMES):
            self.class_combo.addItem(name, i)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        tl.addWidget(self.class_combo)

        self.delete_btn = QPushButton("🗑 Delete (Del)")
        self.delete_btn.setObjectName("DangerButton")
        self.delete_btn.clicked.connect(self._delete_selected)
        tl.addWidget(self.delete_btn)

        tl.addStretch()

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #b8bcc8;")
        tl.addWidget(self.info_label)

        tl.addStretch()

        self.fit_btn = QPushButton("⤢ Fit")
        self.fit_btn.clicked.connect(self.fit_image)
        tl.addWidget(self.fit_btn)

        self.save_btn = QPushButton("Save (Ctrl+S)")
        self.save_btn.setObjectName("SuccessButton")
        self.save_btn.clicked.connect(self.save)
        tl.addWidget(self.save_btn)

        # Scene + view
        self.scene = EditorScene(self)
        self.view = ZoomGraphicsView(self.scene, self)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scene.box_added.connect(self._on_box_added)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self.view)

    def _wire_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self._delete_selected)
        QShortcut(QKeySequence("Right"), self, activated=self.next_image)
        QShortcut(QKeySequence("Left"), self, activated=self.prev_image)
        QShortcut(QKeySequence("A"), self, activated=lambda: self.add_btn.toggle())
        QShortcut(QKeySequence("Escape"), self, activated=lambda: self.add_btn.setChecked(False))

    # --- public API ---
    def open_folder(self, folder: str):
        self._folder = Path(folder)
        self._images = list_images(self._folder)

    def set_image_list(self, images: list[Path]):
        """Restrict ← → navigation to this ordered subset (e.g. current filter tab)."""
        self._images = list(images)

    def open_image(self, image_path: str):
        if self._folder is None:
            self._folder = Path(image_path).parent
            self._images = list_images(self._folder)
        target = Path(image_path)
        try:
            self._index = self._images.index(target)
        except ValueError:
            self._images.append(target)
            self._index = len(self._images) - 1
        self._load_current()

    def next_image(self):
        if not self._images:
            return
        if not self._maybe_save_dirty():
            return
        self._index = (self._index + 1) % len(self._images)
        self._load_current()

    def prev_image(self):
        if not self._images:
            return
        if not self._maybe_save_dirty():
            return
        self._index = (self._index - 1) % len(self._images)
        self._load_current()

    def fit_image(self):
        if not self.scene.sceneRect().isNull():
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # --- internals ---
    def _load_current(self):
        if not self._images:
            return
        path = self._images[self._index]
        self._current_path = path

        reader = QImageReader(str(path))
        size = reader.size()
        if size.isValid():
            self._img_w, self._img_h = size.width(), size.height()

        pix = QPixmap(str(path))
        self.scene.clear()
        if pix.isNull():
            self.info_label.setText(f"[{self._index+1}/{len(self._images)}] could not load: {path.name}")
            return

        self._img_w, self._img_h = pix.width(), pix.height()
        bg = QGraphicsPixmapItem(pix)
        bg.setZValue(-10)
        self.scene.addItem(bg)
        self.scene.setSceneRect(QRectF(0, 0, self._img_w, self._img_h))
        self.scene.set_image_rect(QRectF(0, 0, self._img_w, self._img_h))

        boxes = read_yolo_labels(label_path_for(path), self._img_w, self._img_h)
        for b in boxes:
            r = QRectF(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1)
            item = BoxItem(r, b.cls, conf=b.conf)
            self.scene.addItem(item)

        self._dirty = False
        self.add_btn.setChecked(False)
        self.fit_image()
        self._update_info()

    def _update_info(self):
        n = sum(1 for it in self.scene.items() if isinstance(it, BoxItem))
        name = self._current_path.name if self._current_path else ""
        suffix = "  ●" if self._dirty else ""
        self.info_label.setText(f"[{self._index+1}/{len(self._images)}] {name}  ·  {n} box(es){suffix}")

    def _on_add_toggled(self, on: bool):
        cls = self.class_combo.currentData() or 0
        self.scene.set_add_mode(on, cls)

    def _on_class_changed(self, _ix: int):
        cls = self.class_combo.currentData() or 0
        # Apply to all selected boxes.
        changed = False
        for it in self.scene.selectedItems():
            if isinstance(it, BoxItem) and it.cls != cls:
                it.set_class(cls)
                changed = True
        if changed:
            self._mark_dirty()
        if self.add_btn.isChecked():
            self.scene.set_add_mode(True, cls)

    def _on_box_added(self, _box: BoxItem):
        self._mark_dirty()
        # Stay in add mode for repeated adds; toggle off explicitly with Esc/A.

    def _delete_selected(self):
        any_deleted = False
        for it in list(self.scene.selectedItems()):
            if isinstance(it, BoxItem):
                self.scene.removeItem(it)
                any_deleted = True
        if any_deleted:
            self._mark_dirty()
            self._update_info()

    def _mark_dirty(self):
        self._dirty = True
        self._update_info()

    def _maybe_save_dirty(self) -> bool:
        if not self._dirty:
            return True
        # Auto-save without prompting for snappy navigation.
        return self.save()

    def save(self) -> bool:
        if self._current_path is None:
            return True
        boxes: list[Box] = []
        for it in self.scene.items():
            if isinstance(it, BoxItem):
                it.commit_position()
                r = it.rect()
                # Clip to image
                x1 = max(0.0, min(self._img_w, r.left()))
                y1 = max(0.0, min(self._img_h, r.top()))
                x2 = max(0.0, min(self._img_w, r.right()))
                y2 = max(0.0, min(self._img_h, r.bottom()))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                boxes.append(Box(it.cls, x1, y1, x2, y2, conf=it.conf))

        write_yolo_labels(label_path_for(self._current_path), boxes, self._img_w, self._img_h, include_conf=True)

        # Mark in folder meta
        meta = FolderMeta.load(self._folder)
        meta.mark_edited(self._current_path.name, len(boxes))
        # Recompute conf stats from current boxes.
        m = meta.images[self._current_path.name]
        confs = [b.conf for b in boxes]
        m.max_conf = max(confs) if confs else 0.0
        m.mean_conf = (sum(confs) / len(confs)) if confs else 0.0
        meta.save(self._folder)

        self._dirty = False
        self._update_info()
        self.saved.emit(str(self._current_path), len(boxes))
        return True

    def _on_back(self):
        if not self._maybe_save_dirty():
            return
        self.back_requested.emit()
