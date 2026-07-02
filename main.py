from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication


def main() -> int:
    # When launched via `python main.py` from inside the package dir,
    # add parent so `desktop_app.ui...` imports resolve.
    here = Path(__file__).resolve().parent
    if str(here.parent) not in sys.path:
        sys.path.insert(0, str(here.parent))

    from MastCellDetector.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Mast Cell Detector")
    app.setOrganizationName("Florian Vollmer - ZHAW LSFM")

    qss = here / "ui" / "style.qss"
    if qss.exists():
        app.setStyleSheet(qss.read_text(encoding="utf-8"))

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
