from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

# Accepts either a folder URL, a file URL, or a bare ID
_FOLDER_RE = re.compile(r"drive\.google\.com/.*folders/([A-Za-z0-9_-]+)")
_FILE_RE = re.compile(r"drive\.google\.com/.*[?&]id=([A-Za-z0-9_-]+)")
_FILE_PATH_RE = re.compile(r"drive\.google\.com/file/d/([A-Za-z0-9_-]+)")
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")


def parse_drive_link(link: str) -> tuple[str, str] | None:
    """Returns (kind, id) where kind is 'folder' or 'file', or None if unrecognised."""
    link = link.strip()
    if not link:
        return None
    m = _FOLDER_RE.search(link)
    if m:
        return ("folder", m.group(1))
    m = _FILE_PATH_RE.search(link)
    if m:
        return ("file", m.group(1))
    m = _FILE_RE.search(link)
    if m:
        return ("file", m.group(1))
    if _BARE_ID_RE.match(link):
        return ("folder", link)
    return None


class DriveDownloadWorker(QObject):
    progress = Signal(str)        # status message
    finished = Signal(str)        # local folder path containing downloaded data
    error = Signal(str)

    def __init__(self, link: str, dest_dir: str):
        super().__init__()
        self.link = link
        self.dest_dir = dest_dir
        self._cancel = False

    @Slot()
    def cancel(self):
        self._cancel = True

    @Slot()
    def run(self):
        parsed = parse_drive_link(self.link)
        if not parsed:
            self.error.emit("Could not parse Drive link. Use a folder URL, file URL, or share ID.")
            return

        try:
            import gdown
        except ImportError:
            self.error.emit("gdown not installed. Run: pip install gdown")
            return

        kind, drive_id = parsed
        dest = Path(self.dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        try:
            self.progress.emit(f"Downloading {kind} {drive_id}...")
            if kind == "folder":
                gdown.download_folder(
                    id=drive_id,
                    output=str(dest),
                    quiet=False,
                    use_cookies=False,
                )
                local = dest
            else:
                out_file = dest / f"{drive_id}.bin"
                gdown.download(
                    id=drive_id,
                    output=str(out_file),
                    quiet=False,
                )
                local = dest

            self.finished.emit(str(local))
        except Exception as e:
            self.error.emit(f"Download failed: {e}")
