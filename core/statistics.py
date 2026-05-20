"""Aggregate per-image YOLO detections into a patient-level summary.

The clinically meaningful output of the pipeline is the ratio of Atypisch
to total mast cells per patient. The WHO uses > 25 % Atypisch as a *minor*
criterion for systemic mastocytosis (SM); this is one of several minor
criteria, never a stand-alone diagnosis. We compute the ratio plus
supporting numbers (counts, confidence distribution, contributing image
list) so a clinician can interpret the model output in context.

Inputs are read from the on-disk YOLO label files written by inference
and possibly modified by the user via the annotation editor. We never
mutate them — this module is read-only.
"""

from __future__ import annotations

import statistics as pystats
from dataclasses import dataclass, field
from pathlib import Path

from .annotations import (
    CLASS_NAMES,
    FolderMeta,
    label_path_for,
    list_images,
)

# WHO 2022 minor criterion for SM: >25% Atypisch among mast cells in
# bone marrow biopsy. Threshold is held as a constant so any future
# revision is a one-line change.
WHO_ATYPISCH_THRESHOLD = 0.25

# Below this many total cells we don't render a verdict — the ratio is
# too noisy to interpret. Picked conservatively; clinicians can disagree
# but a hard floor prevents the UI from saying "SM indicated" off 1 cell.
MIN_TOTAL_CELLS_FOR_VERDICT = 10


@dataclass
class ImageStat:
    """Per-image cell counts derived from the on-disk label file."""
    name: str
    n_atypisch: int = 0
    n_normal: int = 0
    confs_atypisch: list[float] = field(default_factory=list)
    confs_normal: list[float] = field(default_factory=list)
    edited: bool = False

    @property
    def n_total(self) -> int:
        return self.n_atypisch + self.n_normal


@dataclass
class FolderStats:
    """Patient-level aggregate. All counts are over labelled images only."""
    folder: str = ""
    n_images: int = 0
    n_images_with_cells: int = 0
    n_atypisch: int = 0
    n_normal: int = 0
    per_image: list[ImageStat] = field(default_factory=list)
    n_edited: int = 0

    @property
    def n_total(self) -> int:
        return self.n_atypisch + self.n_normal

    @property
    def atypisch_ratio(self) -> float:
        """Atypisch / (Atypisch + Normal). Returns 0.0 if no cells found."""
        return self.n_atypisch / self.n_total if self.n_total > 0 else 0.0

    @property
    def has_enough_data(self) -> bool:
        return self.n_total >= MIN_TOTAL_CELLS_FOR_VERDICT

    def verdict(self) -> tuple[str, str]:
        """Return (label, severity) where severity ∈ {'sm', 'no_sm', 'insufficient'}.

        The label is human-readable; severity is what the UI uses to colour
        the banner. Never delivers a diagnosis — only a flag.
        """
        if not self.has_enough_data:
            return (
                f"Insufficient data ({self.n_total} mast cells detected; "
                f"need ≥{MIN_TOTAL_CELLS_FOR_VERDICT} for an indication)",
                "insufficient",
            )
        ratio_pct = self.atypisch_ratio * 100
        threshold_pct = WHO_ATYPISCH_THRESHOLD * 100
        if self.atypisch_ratio > WHO_ATYPISCH_THRESHOLD:
            return (
                f"Indication of SM — Atypisch fraction {ratio_pct:.1f}% "
                f"exceeds WHO minor criterion (>{threshold_pct:.0f}%)",
                "sm",
            )
        return (
            f"No SM indication — Atypisch fraction {ratio_pct:.1f}% "
            f"below WHO minor criterion ({threshold_pct:.0f}%)",
            "no_sm",
        )

    def confidence_summary(self, cls: int) -> dict | None:
        """Return mean / median / min / max for one class's confidences."""
        confs: list[float] = []
        for img in self.per_image:
            confs.extend(img.confs_atypisch if cls == 0 else img.confs_normal)
        if not confs:
            return None
        return {
            "n":      len(confs),
            "mean":   pystats.fmean(confs),
            "median": pystats.median(confs),
            "min":    min(confs),
            "max":    max(confs),
        }


def _parse_label_file(p: Path) -> tuple[list[float], list[float]]:
    """Return (atypisch_confs, normal_confs) for one label file.

    YOLO labels are: `cls cx cy w h [conf]`. Confidence is optional and
    only present in files written by inference (we set include_conf=True).
    User-drawn boxes have conf=1.0.
    """
    atyp, norm = [], []
    if not p.exists() or p.stat().st_size == 0:
        return atyp, norm
    with open(p) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                conf = float(parts[5]) if len(parts) >= 6 else 1.0
            except ValueError:
                continue
            if cls == 0:
                atyp.append(conf)
            elif cls == 1:
                norm.append(conf)
    return atyp, norm


def compute_folder_stats(folder: str | Path) -> FolderStats:
    """Walk a folder, read every label file, aggregate into FolderStats."""
    folder = Path(folder)
    stats = FolderStats(folder=str(folder))
    if not folder.exists():
        return stats

    meta = FolderMeta.load(folder)
    images = list_images(folder)
    stats.n_images = len(images)

    for img in images:
        lbl = label_path_for(img)
        atyp_confs, norm_confs = _parse_label_file(lbl)
        im_meta = meta.images.get(img.name)
        edited = bool(im_meta and im_meta.edited)

        s = ImageStat(
            name=img.name,
            n_atypisch=len(atyp_confs),
            n_normal=len(norm_confs),
            confs_atypisch=atyp_confs,
            confs_normal=norm_confs,
            edited=edited,
        )
        stats.per_image.append(s)
        stats.n_atypisch += s.n_atypisch
        stats.n_normal += s.n_normal
        if s.n_total > 0:
            stats.n_images_with_cells += 1
        if edited:
            stats.n_edited += 1

    return stats


def format_summary_text(stats: FolderStats) -> str:
    """Plain-text summary suitable for export (TXT, clipboard, PDF body)."""
    label, _ = stats.verdict()
    lines = [
        "Mast Cell Detection — Patient Summary",
        "=" * 60,
        f"Folder         : {stats.folder}",
        f"Images scanned : {stats.n_images}",
        f"Images with    : {stats.n_images_with_cells}",
        f"  detections",
        f"User-edited    : {stats.n_edited}",
        "",
        f"Atypisch ({CLASS_NAMES[0]}) : {stats.n_atypisch}",
        f"Normal   ({CLASS_NAMES[1]}) : {stats.n_normal}",
        f"Total            : {stats.n_total}",
        "",
        f"Atypisch fraction: {stats.atypisch_ratio * 100:.2f}%",
        f"WHO threshold    : {WHO_ATYPISCH_THRESHOLD * 100:.0f}% (minor criterion for SM)",
        "",
        "Verdict:",
        f"  {label}",
        "",
    ]

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cs = stats.confidence_summary(cls_idx)
        if cs:
            lines.append(
                f"{cls_name} confidence — n={cs['n']}, "
                f"mean={cs['mean']:.3f}, median={cs['median']:.3f}, "
                f"range=[{cs['min']:.3f}, {cs['max']:.3f}]"
            )
    lines.append("")
    lines.append(
        "Disclaimer: This output is a research-prototype indication based "
        "on a single WHO minor criterion (Atypisch fraction). It is not a "
        "diagnosis and must be interpreted by a qualified clinician in "
        "the context of the full diagnostic workup."
    )
    return "\n".join(lines)
