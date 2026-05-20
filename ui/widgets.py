from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QSize,
    Qt,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QBrush, QFont
from PySide6.QtWidgets import QListView, QStyledItemDelegate, QStyle

from ..core.annotations import (
    CLASS_COLORS,
    Box,
    FolderMeta,
    ImageMeta,
    label_path_for,
    read_yolo_labels,
)


THUMB_SIZE = 192
CACHE_LIMIT = 600


class ThumbnailCache(QObject):
    """LRU cache of thumbnails generated in a thread pool.

    Workers produce QImage (thread-safe). The QPixmap is built lazily on the
    main thread via get(), since QPixmap requires the GUI thread.
    Emits `ready` when an image finishes loading so views can refresh.
    """

    ready = Signal(str)  # image_path

    def __init__(self, max_workers: int = 4, parent=None):
        super().__init__(parent)
        self._img_cache: "OrderedDict[str, QImage]" = OrderedDict()
        self._pix_cache: "OrderedDict[str, QPixmap]" = OrderedDict()
        self._pending: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def get(self, path: str) -> Optional[QPixmap]:
        pix = self._pix_cache.get(path)
        if pix is not None:
            self._pix_cache.move_to_end(path)
            return pix
        img = self._img_cache.get(path)
        if img is None:
            return None
        # Lazy convert on the main thread.
        pix = QPixmap.fromImage(img)
        self._pix_cache[path] = pix
        if len(self._pix_cache) > CACHE_LIMIT:
            self._pix_cache.popitem(last=False)
        return pix

    def request(self, path: str, boxes: list[Box]) -> Optional[QPixmap]:
        pix = self.get(path)
        if pix is not None:
            return pix
        if path in self._pending:
            return None
        self._pending.add(path)
        self._executor.submit(self._load, path, boxes)
        return None

    def _load(self, path: str, boxes: list[Box]):
        try:
            img = QImage(path)
            if img.isNull():
                return
            scaled = img.scaled(
                THUMB_SIZE, THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            sx = scaled.width() / max(img.width(), 1)
            sy = scaled.height() / max(img.height(), 1)
            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            for b in boxes:
                col = CLASS_COLORS[b.cls % len(CLASS_COLORS)]
                pen = QPen(QColor(*col, 220))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawRect(int(b.x1 * sx), int(b.y1 * sy),
                                 int((b.x2 - b.x1) * sx), int((b.y2 - b.y1) * sy))
            painter.end()
            self._store(path, scaled)
        except Exception:
            pass
        finally:
            self._pending.discard(path)

    def _store(self, path: str, img: QImage):
        self._img_cache[path] = img
        if len(self._img_cache) > CACHE_LIMIT:
            self._img_cache.popitem(last=False)
        self.ready.emit(path)

    def invalidate(self, path: str):
        self._img_cache.pop(path, None)
        self._pix_cache.pop(path, None)

    def clear(self):
        self._img_cache.clear()
        self._pix_cache.clear()


class GalleryModel(QAbstractListModel):
    """Model that holds (image_path, ImageMeta) pairs and provides thumbnails."""

    PathRole = Qt.UserRole + 1
    MetaRole = Qt.UserRole + 2

    def __init__(self, cache: ThumbnailCache, parent=None):
        super().__init__(parent)
        self._items: list[tuple[Path, ImageMeta]] = []
        self._cache = cache
        self._cache.ready.connect(self._on_thumb_ready)
        self._row_for_path: dict[str, int] = {}

    def set_items(self, items: list[tuple[Path, ImageMeta]]):
        self.beginResetModel()
        self._items = items
        self._row_for_path = {str(p): i for i, (p, _) in enumerate(items)}
        self.endResetModel()

    def update_meta(self, path: str, meta: ImageMeta):
        row = self._row_for_path.get(path)
        if row is None:
            return
        p, _ = self._items[row]
        self._items[row] = (p, meta)
        self._cache.invalidate(path)
        idx = self.index(row)
        self.dataChanged.emit(idx, idx)

    def image_paths(self) -> list[Path]:
        return [p for p, _ in self._items]

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._items):
            return None
        path, meta = self._items[index.row()]
        if role == Qt.DisplayRole:
            return path.name
        if role == Qt.DecorationRole:
            # Fast path: cache hit, no disk read.
            cached = self._cache.get(str(path))
            if cached is not None:
                return cached
            from PySide6.QtGui import QImageReader
            reader = QImageReader(str(path))
            size = reader.size()
            if not size.isValid() or size.width() <= 0:
                return None
            boxes = read_yolo_labels(label_path_for(path), size.width(), size.height())
            return self._cache.request(str(path), boxes)
        if role == self.PathRole:
            return str(path)
        if role == self.MetaRole:
            return meta
        return None

    @Slot(str)
    def _on_thumb_ready(self, path: str):
        row = self._row_for_path.get(path)
        if row is None:
            return
        idx = self.index(row)
        self.dataChanged.emit(idx, idx, [Qt.DecorationRole])


class GalleryDelegate(QStyledItemDelegate):
    """Draws thumbnail + name + status badge."""

    def sizeHint(self, option, index):
        return QSize(THUMB_SIZE + 16, THUMB_SIZE + 44)

    def paint(self, painter: QPainter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = option.rect.adjusted(6, 6, -6, -6)
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, QColor("#2a3a55"))
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(option.rect, QColor("#353945"))
        else:
            painter.fillRect(option.rect, QColor("#2a2d36"))

        pix = index.data(Qt.DecorationRole)
        thumb_rect = rect.adjusted(0, 0, 0, -32)
        if isinstance(pix, QPixmap) and not pix.isNull():
            scaled = pix.scaled(thumb_rect.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            x = thumb_rect.x() + (thumb_rect.width() - scaled.width()) // 2
            y = thumb_rect.y() + (thumb_rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor("#5a5e6a"))
            painter.drawText(thumb_rect, Qt.AlignmentFlag.AlignCenter, "loading…")

        # Name
        name = index.data(Qt.DisplayRole) or ""
        painter.setPen(QColor("#e6e8ee"))
        font = QFont(painter.font())
        font.setPointSize(9)
        painter.setFont(font)
        name_rect = rect.adjusted(0, thumb_rect.height(), 0, -16)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignCenter,
                         _elide(name, 28))

        # Status badge
        meta: ImageMeta = index.data(GalleryModel.MetaRole)
        if meta is not None:
            badge_text, badge_color = _status_badge(meta)
            badge_rect = rect.adjusted(0, rect.height() - 16, 0, 0)
            painter.setPen(badge_color)
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

        painter.restore()


def _elide(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _status_badge(meta: ImageMeta) -> tuple[str, QColor]:
    if meta.edited:
        return ("✎ edited", QColor("#f0a040"))
    if meta.n_boxes == 0:
        return ("• no detection", QColor("#8a90a0"))
    if meta.max_conf < 0.4:
        return (f"⚠ {meta.n_boxes} box · low conf", QColor("#d65a5a"))
    if meta.max_conf < 0.8:
        return (f"● {meta.n_boxes} box · {meta.max_conf:.2f}", QColor("#f0c040"))
    return (f"● {meta.n_boxes} box · {meta.max_conf:.2f}", QColor("#3fb46a"))


class GalleryView(QListView):
    image_activated = Signal(str)
    context_menu_requested = Signal(str)  # image_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setMovement(QListView.Movement.Static)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setSpacing(4)
        self.setUniformItemSizes(True)
        self.setMouseTracking(True)
        self.activated.connect(self._on_activate)
        self.doubleClicked.connect(self._on_activate)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def _on_activate(self, index: QModelIndex):
        path = index.data(GalleryModel.PathRole)
        if path:
            self.image_activated.emit(path)

    def _on_context_menu(self, pos):
        index = self.indexAt(pos)
        if not index.isValid():
            return
        path = index.data(GalleryModel.PathRole)
        if path:
            self.context_menu_requested.emit(path)
