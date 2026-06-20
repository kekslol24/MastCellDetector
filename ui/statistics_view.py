"""Patient-summary view — verdict banner, counts table, per-image breakdown.

Sits in the main window's QStackedWidget alongside the gallery and the
annotation editor. Refreshes lazily — `refresh(folder)` is called by the
main window after inference completes or after a user edit is saved.

The view is intentionally *informational only*. It never triggers
inference, training, or any model action. The intent is that a clinician
runs detection, optionally corrects mistakes, then opens this view to
read the WHO-criterion ratio and decide whether to escalate.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.annotations import CLASS_NAMES
from ..core.statistics import (
    FolderStats,
    WHO_ATYPISCH_THRESHOLD,
    compute_folder_stats,
    format_summary_text,
)


# Banner colours by severity. Keep these in one place so the styling is
# consistent if we later support theming.
_SEVERITY_STYLE = {
    "sm":           "background:#b8001f; color:white;",
    "no_sm":        "background:#1f7a3a; color:white;",
    "insufficient": "background:#7a7a7a; color:white;",
}


class StatisticsView(QWidget):
    """Read-only summary panel computed from on-disk labels + folder meta."""

    # Emitted when the user clicks "Open image in editor" in the per-image
    # table. The main window listens and switches to the editor stack item.
    open_image_requested = Signal(str)   # absolute image path

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._folder: Path | None = None
        self._stats: FolderStats | None = None
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(14)

        title = QLabel("Patient Summary")
        title.setObjectName("StatsTitle")
        f = QFont()
        f.setPointSize(18)
        f.setBold(True)
        title.setFont(f)
        outer.addWidget(title)

        self._verdict = QLabel("Run inference on a folder to see results.")
        self._verdict.setWordWrap(True)
        self._verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verdict.setMinimumHeight(72)
        vf = QFont()
        vf.setPointSize(14)
        vf.setBold(True)
        self._verdict.setFont(vf)
        self._verdict.setStyleSheet(_SEVERITY_STYLE["insufficient"])
        outer.addWidget(self._verdict)

        # Counts table (small, fixed shape).
        counts_row = QHBoxLayout()
        counts_row.setSpacing(20)
        self._counts_label = QLabel("")
        self._counts_label.setTextFormat(Qt.TextFormat.RichText)
        self._counts_label.setWordWrap(True)
        counts_row.addWidget(self._counts_label, stretch=2)

        self._conf_label = QLabel("")
        self._conf_label.setTextFormat(Qt.TextFormat.RichText)
        self._conf_label.setWordWrap(True)
        counts_row.addWidget(self._conf_label, stretch=1)
        outer.addLayout(counts_row)

        # WHO caveat box — non-dismissable, always present.
        caveat = QLabel(
            "<b>Caveat.</b> The Atypisch fraction is one of several WHO "
            "<i>minor</i> criteria for systemic mastocytosis. This panel does "
            "not provide a diagnosis. Interpretation must be made by a "
            "qualified clinician in the context of bone marrow histology, "
            "tryptase levels, and KIT mutation testing. Model recall is "
            "imperfect — undetected cells inflate the Atypisch fraction "
            "when Normal cells are missed and deflate it when Atypisch "
            "cells are missed."
        )
        caveat.setWordWrap(True)
        caveat.setObjectName("InfoLabel")
        caveat.setStyleSheet(
            "background:#222; color:#ddd; padding:10px; border-radius:4px;"
        )
        outer.addWidget(caveat)

        # Per-image breakdown.
        ph = QLabel("Per-image breakdown")
        phf = QFont()
        phf.setBold(True)
        ph.setFont(phf)
        outer.addWidget(ph)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Image", "Atypisch", "Normal", "Total", "Edited"]
        )
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for col in range(1, 5):
            self._table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        outer.addWidget(self._table, stretch=1)

        # Actions
        action_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip(
            "Re-read all label files from disk. Run automatically after "
            "inference and after every edit."
        )
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        action_row.addWidget(self._refresh_btn)

        self._export_btn = QPushButton("Export summary…")
        self._export_btn.setToolTip("Save the verdict + counts as a .txt file.")
        self._export_btn.clicked.connect(self._on_export)
        action_row.addWidget(self._export_btn)

        action_row.addStretch()
        outer.addLayout(action_row)

    # ---------------------------------------------------------- public API
    def refresh(self, folder: str | Path | None):
        """Recompute and redraw. Pass None to clear."""
        if folder is None:
            self._folder = None
            self._stats = None
            self._verdict.setText("No folder loaded.")
            self._verdict.setStyleSheet(_SEVERITY_STYLE["insufficient"])
            self._counts_label.setText("")
            self._conf_label.setText("")
            self._table.setRowCount(0)
            return

        self._folder = Path(folder)
        self._stats = compute_folder_stats(self._folder)
        self._render(self._stats)

    # --------------------------------------------------------- internals
    def _render(self, stats: FolderStats):
        # Verdict banner
        label, severity = stats.verdict()
        self._verdict.setText(label)
        self._verdict.setStyleSheet(_SEVERITY_STYLE.get(
            severity, _SEVERITY_STYLE["insufficient"]
        ))

        threshold_pct = WHO_ATYPISCH_THRESHOLD * 100
        ratio_pct = stats.atypisch_ratio * 100
        self._counts_label.setText(
            f"<b>Counts</b><br>"
            f"Images scanned: {stats.n_images}<br>"
            f"With detections: {stats.n_images_with_cells}<br>"
            f"User-edited: {stats.n_edited}<br><br>"
            f"<b>Atypisch:</b> {stats.n_atypisch}<br>"
            f"<b>Normal:</b> {stats.n_normal}<br>"
            f"<b>Total mast cells:</b> {stats.n_total}<br><br>"
            f"<b>Atypisch fraction:</b> {ratio_pct:.2f}%<br>"
            f"<b>WHO threshold:</b> {threshold_pct:.0f}%"
        )

        conf_lines = ["<b>Confidence summary</b>"]
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cs = stats.confidence_summary(cls_idx)
            if cs:
                conf_lines.append(
                    f"<br><b>{cls_name}</b> (n={cs['n']})"
                    f"<br>&nbsp;&nbsp;mean: {cs['mean']:.3f}"
                    f"<br>&nbsp;&nbsp;median: {cs['median']:.3f}"
                    f"<br>&nbsp;&nbsp;range: [{cs['min']:.3f}, {cs['max']:.3f}]"
                )
            else:
                conf_lines.append(
                    f"<br><b>{cls_name}</b>: no detections"
                )
        self._conf_label.setText("".join(conf_lines))

        # Per-image table
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(stats.per_image))
        for row, im in enumerate(stats.per_image):
            self._table.setItem(row, 0, QTableWidgetItem(im.name))

            for col, value in (
                (1, im.n_atypisch),
                (2, im.n_normal),
                (3, im.n_total),
            ):
                item = QTableWidgetItem()
                item.setData(Qt.ItemDataRole.DisplayRole, int(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row, col, item)

            edited_item = QTableWidgetItem("✓" if im.edited else "")
            edited_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 4, edited_item)
        self._table.setSortingEnabled(True)
        # Initial sort: most cells first so suspicious images surface.
        self._table.sortItems(3, Qt.SortOrder.DescendingOrder)

    def current_image_paths(self, folder: Path) -> list[Path]:
        """Return image paths in the table's current display order (respects sort)."""
        paths = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is None:
                continue
            p = folder / item.text()
            if p.exists():
                paths.append(p)
        return paths

    # ---------------------------------------------------------- handlers
    def _on_refresh_clicked(self):
        if self._folder is not None:
            self.refresh(self._folder)

    def _on_row_double_clicked(self, row: int, _col: int):
        if not self._folder:
            return
        item = self._table.item(row, 0)
        if item is None:
            return
        path = self._folder / item.text()
        if path.exists():
            self.open_image_requested.emit(str(path))

    def _on_export(self):
        if self._stats is None or self._folder is None:
            QMessageBox.information(
                self, "Nothing to export",
                "Load a folder and run inference first."
            )
            return
        default = (
            self._folder / f"summary_{self._folder.name}.txt"
            if self._folder
            else Path.home() / "summary.txt"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save summary", str(default),
            "Text files (*.txt);;All files (*.*)"
        )
        if not path:
            return
        try:
            Path(path).write_text(format_summary_text(self._stats), encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Saved", f"Summary written to:\n{path}")
