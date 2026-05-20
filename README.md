# Mast Cell Detector — Desktop

A native desktop replacement for the Gradio app. Designed to handle 20k+ image
runs that the browser-based version cannot.


## Features

- **Source**: open a local folder or paste a Google Drive folder/file link
- **Auto hardware**: detects GPU + VRAM and recommends batch size; falls back to CPU
- **Streaming inference**: writes YOLO-format `.txt` next to each image as results stream in
- **Gallery view**: lazy-loaded thumbnails with overlaid boxes; filter by
  All / With detections / No detection / Low confidence / Atypisch / Normal / Edited
- **Annotation editor**: click a thumbnail to edit
  - Drag boxes to move, drag corner handles to resize
  - `Del` deletes selected box(es)
  - `A` toggles "draw new box" mode; class chosen from dropdown
  - `Ctrl+S` saves; `←` / `→` navigates with auto-save
  - Mouse wheel = zoom, middle mouse = pan
- **IoU NMS control**: sidebar slider adjusts the non-maximum suppression threshold so overlapping boxes (lower confidence one) are removed at the desired strictness.

## Install

```bash
cd desktop_app
pip install -r requirements.txt
```

`requirements.txt` pulls a generic `torch` wheel. If you want a specific
backend, install torch first (then `pip install -r requirements.txt` will
skip it):

| Backend | Command |
|---------|---------|
| **NVIDIA (CUDA 12.1)** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| **AMD (ROCm 6.1, Linux)** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1` |
| **Apple Silicon (MPS)** | `pip install torch torchvision`  (the macOS wheel ships with MPS) |
| **CPU only** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

The hardware panel in the sidebar shows which backend was detected
(NVIDIA / AMD / Apple / CPU). PyTorch-ROCm reuses the CUDA namespace, so
internally the device string is still `cuda:0` — Ultralytics handles this
transparently.

ROCm notes:

- Linux only (no official ROCm-PyTorch on Windows yet)
- Supported AMD cards are the Instinct MI series and most Radeon RX 7000 / 6000.
  RDNA2/3 cards may need `HSA_OVERRIDE_GFX_VERSION=10.3.0` (or similar) before
  launching the app — check the [ROCm support matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

## Run

```bash
# from BA/
python -m desktop_app.main

# or directly
cd desktop_app
python main.py
```

The app remembers nothing across sessions; it stores per-folder state in
`<folder>/.dapp_meta.json` (inference timestamp, per-image conf stats, edited flag).

## File layout written to disk

For a folder with 20k images:

```
your_folder/
├── img_001.jpg
├── img_001.txt           # YOLO-format labels (cls cx cy w h conf)
├── img_002.jpg
├── img_002.txt           # empty file = confirmed negative
├── ...
├── .dapp_meta.json       # per-image stats + edited flag
└── .retrain_runs/        # created when you click Retrain
    └── retrain_<ts>/
        └── weights/
            └── best.pt
```

Edited labels are never overwritten by re-running inference (the `edited` flag in
`.dapp_meta.json` protects them).

## Building a Windows .exe

```bash
cd desktop_app
pip install pyinstaller
pyinstaller --noconfirm --windowed --name "MastCellDetector" \
    --exclude-module PyQt6
    --add-data "dist/ui/style.qss;ui" \
    --collect-data ultralytics \
    main.py
```

Output: `dist/MastCellDetector/MastCellDetector.exe`. Distribute the entire
`dist/MastCellDetector/` folder.

For a single-file .exe (slower start, larger):

```bash
pyinstaller --noconfirm --windowed --onefile --name "MastCellDetector" \
    --exclude-module PyQt6
    --add-data "ui/style.qss;ui" \
    --collect-data ultralytics \
    main.py
```

Bundling CUDA torch significantly inflates the binary; if your target machines
have no GPU, install the CPU-only torch wheel before running PyInstaller:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For an AMD/ROCm distribution, do not try to ship a single Windows .exe — ROCm
PyTorch is Linux-only. Build on Linux instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
pyinstaller --noconfirm --windowed --name "MastCellDetector" \
    --add-data "ui/style.qss:ui" \
    --collect-data ultralytics \
    main.py
```

Note the `:` (Linux) vs `;` (Windows) separator in `--add-data`.

## Default model path

The app probes for weights in this order on startup:

1. `BA/hpc/DL_Modell_FV.pt`
2. `BA/app/DL_Modell_FV.pt`
3. `yolo11n.pt` (Ultralytics will auto-download)

Use **Browse…** in the sidebar to pick a different `.pt` file.

## Workflow

1. **Open Folder** → load 20k images
2. **Run Inference** → progress bar, labels stream to disk
3. Filter to **Low confidence**, **No detection**, **Atypisch**, or **Normal** to find suspicious cases
4. Click an image → **Annotation Editor**
   - Delete a wrong box (false positive)
   - Drag corner to resize, drag body to move
   - Press `A`, click+drag to draw a new box; pick class from dropdown
   - `Ctrl+S` (or just press → for next; auto-saves)
5. Switch to **Statistics** tab to see the patient-level Atypisch fraction vs WHO 25% threshold

## Limitations / known issues

- Drive folders behind login or rate-limits will fail; use folders shared "Anyone with link"
- Atypisch / Normal gallery filters only populate after a fresh inference run (they read from `.dapp_meta.json` counts written by the worker)
