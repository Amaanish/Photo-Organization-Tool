# Intelligent Photo Organization Tool

A powerful face recognition tool that helps you quickly locate and organize photos containing specific people from large image collections. Built with InsightFace and OpenCV for accurate, fast facial recognition — now with **split CPU/GPU execution** using Apple Metal (CoreML) for maximum performance on macOS.

## ⚠️ IMPORTANT: Designed exclusively for macOS. Uses AppleScript for system integration.

---

## What It Does

Scans through folders of images and automatically identifies photos containing specific people you're searching for. Once identified, it selects those images directly in Finder — making it easy to organize, copy, or manage your photo collections.

**New in this version:**
- Detection runs on **CPU** for stability
- Recognition runs on **GPU via CoreML (Apple Metal)** for speed
- Search for **multiple people** at once
- Improved embedding normalization for better accuracy

---

## Platform Requirements & Warnings

### macOS Only
- **This tool ONLY works on macOS** due to its heavy reliance on AppleScript
- Requires macOS with AppleScript and CoreML support
- Will **not** function on Windows or Linux

### Privacy & Ethics Notice
- **This tool processes facial biometric data**
- Only use on images you own or have explicit permission to process
- Be mindful of privacy laws in your jurisdiction (GDPR, CCPA, etc.)
- Consider informing people if you're processing their images
- **Do not use for surveillance or unauthorized identification**

### System Permissions Required
- Full disk access for reading image folders
- Finder automation permissions for file selection
- CoreML/Metal GPU access (granted automatically on macOS)

---

## Installation

### Prerequisites
```bash
pip install opencv-python insightface numpy onnxruntime
```

### GPU / Metal Support (Recommended for Apple Silicon & Intel Macs)
```bash
pip install onnxruntime-coreml
```
> CoreML execution is configured automatically — no extra setup needed once installed.

### Model Download
InsightFace will auto-download the `buffalo_l` models on first run. They'll be saved to:
```
~/.insightface/models/buffalo_l/
├── det_10g.onnx       ← Detection (runs on CPU)
└── w600k_r50.onnx     ← Recognition (runs on GPU via CoreML)
```

### Directory Structure
```
faceFind.py
training_data/
├── person1_name/
│   ├── photo1.jpg
│   └── photo2.png
├── person2_name/
│   ├── photo1.jpg
│   └── ...
└── ...
```

---

## 📖 Usage

1. **Prepare Training Data**
   - Create a folder inside `training_data/` named after each person
   - Add clear, well-lit photos of their face (5+ recommended)
   - Only one reference image per person is needed at runtime (first valid one is used)

2. **Run the Tool**
   ```bash
   python3 faceFind.py
   ```

3. **Follow the Prompts**
   - Enter the name(s) of the people you want to find
   - Type `n` when asked if you want to add another person to start scanning
   - Select the folder containing your images via the macOS dialog

4. **Review Results**
   - Matched images are automatically selected in Finder
   - From there you can copy, move, tag, or organize them freely

---

## Configuration

### Accuracy Tuning
Modify the `threshold` value in the `faces()` function:

| Threshold | Behaviour |
|-----------|-----------|
| `0.50+` | Very strict — minimal false positives |
| `0.45` | **Default** — balanced accuracy |
| `0.35–0.40` | More lenient — catches harder angles |
| `< 0.35` | Loose — may include similar-looking faces |

### ⚡ Performance Tuning
- Adjust `max_workers` in `ThreadPoolExecutor` based on your CPU core count
- Images wider than 1920px are automatically downscaled before processing
- Default: **6 workers**

---

## Technical Details

### Face Recognition Pipeline
1. **Load Phase** — Extracts and normalizes facial embeddings from your training photos
2. **Detection Phase** — Uses `det_10g.onnx` on CPU to locate faces in each image
3. **Recognition Phase** — Uses `w600k_r50.onnx` on GPU (CoreML) to generate face embeddings
4. **Matching Phase** — Compares embeddings using dot product (cosine similarity on normalized vectors)
5. **Selection Phase** — AppleScript selects all matched files in Finder

### Why Split CPU/GPU?
The detection model benefits from CPU stability, while the recognition model is computationally heavier and gets a significant speedup from Apple's Metal GPU via CoreML — giving you the best of both worlds.

### Dependencies
- `insightface` — State-of-the-art face recognition models
- `opencv-python` — Image processing and resizing
- `numpy` — Embedding math and cosine similarity
- `onnxruntime` — Model inference (CPU + CoreML backend)
- `concurrent.futures` — Multi-threaded image scanning

---

## Troubleshooting

### Common Issues

**"No module named 'insightface'"**
```bash
pip install insightface
```

**CoreML not being used / slow performance**
```bash
pip install onnxruntime-coreml
```
Make sure you're not running in a virtualenv that blocks system frameworks.

**macOS Security Warnings**
Go to System Settings → Privacy & Security and allow Terminal (or your Python environment) to access:
- Files and Folders
- Finder automation

**Poor Recognition Results**
- Add more training photos per person (varied angles, lighting)
- Slightly lower the similarity threshold (e.g. `0.40` → `0.35`)
- Ensure training images have a clearly visible, unobstructed face

**"Can't find name in training data"**
Folder names in `training_data/` must match exactly what you type (case-sensitive).

**No folder selected**
The macOS folder picker appeared but was dismissed — re-run and select a valid folder.

---

## Performance Notes

Processing speed depends on:
- Number and resolution of images in the target folder
- Number of faces per image
- Available CPU cores
- GPU availability via CoreML (Apple Silicon highly recommended)

Typical performance with CoreML enabled: **~5–15 images/second** on modern Apple Silicon.

---

## Contributing

Feel free to open issues or submit pull requests! Areas for improvement:
- Windows/Linux support (swap AppleScript for cross-platform alternatives)
- GUI interface
- Support for video files
- Batch export / auto-copy to output folder

---

## 📄 Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface) for the face recognition models
- OpenCV community for image processing tools

---

**Created by [Amaanish](https://github.com/Amaanish)**
