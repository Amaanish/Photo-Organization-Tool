# Intelligent Photo Organization Tool


A powerful face recognition tool that helps you quickly locate and organize photos containing specific people from large image collections. Built with InsightFace and OpenCV for accurate facial recognition.

## IMPORTANT: This tool is designed specifically for macOS and uses AppleScript for system integration. 

##  What It Does

FaceFind scans through folders of images and automatically identifies photos containing a specific person you're looking for. Once identified, it automatically selects those images in Finder, making it easy to organize, copy, or manage your photo collections.

##  Platform Requirements & Warnings

### macOS Only
- **This tool ONLY works on macOS** due to its heavy reliance on AppleScript
- Requires macOS with AppleScript support
- Will not function on Windows or Linux systems

### Privacy & Ethics Notice
- **This tool processes facial biometric data**
- Only use on images you own or have explicit permission to process
- Be mindful of privacy laws in your jurisdiction (GDPR, CCPA, etc.)
- Consider informing people if you're processing their images
- Do not use for surveillance or unauthorized identification

### System Permissions
- Requires camera/microphone permissions if processing live images
- Needs full disk access for reading image folders
- Finder integration requires automation permissions

## 🚀 Installation

### Prerequisites
```bash
# Install Python dependencies
pip install opencv-python insightface numpy pillow
```

### GPU Support (Optional but Recommended)
For faster processing, install CUDA-compatible versions:
```bash
pip install onnxruntime-gpu
```

### Directory Structure
Create a training data folder structure like this:
```
main.py
training_data/
├── person1_name/
│   ├── photo1.jpg
│   ├── photo2.png
│   └── ...
├── person2_name/
│   ├── photo1.jpg
│   └── ...
└── ...
```

##  Usage

1. **Prepare Training Data**
   - Create folders named after each person
   - Add clear photos of each person's face (5 photos is recommended but more photos the better)
   - Use high-quality, well-lit images for best results

2. **Run the Tool**
   ```bash
   python3 faceFind.py
   ```

3. **Follow the Prompts**
   - Enter the name of the person you want to find
   - Select the folder containing images to search through
   - Wait for processing to complete

4. **Review Results**
   - Found images will be automatically selected in Finder
   - You can then copy, move, or organize them as needed

##  Configuration

### Accuracy Tuning
Modify the `threshold` value in the `faces()` function:
- Higher values (0.4-0.5): More strict matching, fewer false positives
- Lower values (0.25-0.35): More lenient matching, may include similar faces
- Default: 0.35 (balanced accuracy)

### Performance Tuning
- Adjust `max_workers` in ThreadPoolExecutor for your CPU
- Modify `IMG_SIZE` for different speed/accuracy trade-offs
- Default: 8 workers, 224px image size

## 🔧 Technical Details

### Face Recognition Pipeline
1. **Training Phase**: Extracts facial embeddings from known photos
2. **Detection Phase**: Processes unknown images and extracts face embeddings
3. **Matching Phase**: Compares embeddings using cosine similarity
4. **Selection Phase**: Uses AppleScript to select matching files in Finder

### Dependencies
- `insightface`: State-of-the-art face recognition
- `opencv-python`: Image processing
- `numpy`: Numerical computations
- `PIL`: Image handling
- `concurrent.futures`: Multi-threading

##  Troubleshooting

### Common Issues
1. **"No module named 'insightface'"**
   ```bash
   pip install insightface
   ```

2. **macOS Security Warnings**
   - Go to System Preferences > Security & Privacy
   - Allow terminal/Python to access files and Finder

3. **Poor Recognition Results**
   - Add more training photos per person
   - Use higher quality, well-lit training images
   - Adjust the similarity threshold

4. **Slow Performance**
   - Reduce image batch size
   - Lower the IMG_SIZE parameter
   - Ensure you have adequate RAM

### Error Messages
- **"Too many invalid attempts"**: Check person names match folder names exactly
- **"No folder selected"**: Ensure you select a valid folder in the dialog
- **AppleScript errors**: Verify you're running on macOS with proper permissions

##  Performance Notes

- Processing speed depends on:
  - Number of images in target folder
  - Image resolution and file sizes
  - Available CPU cores and RAM
  - GPU availability (if using CUDA)

- Typical performance: ~2-5 images per second on modern hardware

##  Contributing

Feel free to submit issues and enhancement requests! Areas for improvement:
- Cross-platform compatibility
- GUI interface
- Batch processing improvements
- Additional output formats

##  Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for the face recognition models
- OpenCV community for image processing tools

---

**Created by [Amaanish](https://github.com/Amaanish)**

*Remember: With great power comes great responsibility. Use this tool ethically and respect others' privacy.*
