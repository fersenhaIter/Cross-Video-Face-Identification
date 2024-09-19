# Cross-Video Face Identification

<img src="cover_.jpg" alt="Project Cover" align="right" style="width: 200px; border: 2px solid black; margin-left: 20px; float: right;"/>

## Overview

This project processes video footage to detect faces, generate embeddings, and identify recurring individuals across multiple videos. Originally designed for war journalism and evacuations, it has been optimized for accuracy and performance. The system creates individual folders for each detected face, along with logs that track where each face appears across different videos. By using a combination of detection and clustering techniques, the system maximizes both precision and speed.

## Key Features

### Optimized Frame Selection
Advanced techniques avoid redundant frame processing:
1. **Motion Detection**: Frames with minimal motion are skipped using **optical flow**, reducing unnecessary processing.
2. **Brightness Analysis**: Low-light frames are filtered out to avoid processing poor-quality images.

*Combination rationale*: Together, these techniques reduce the processing of irrelevant frames while ensuring important data is preserved, making the system faster without losing accuracy.

### Face Detection
Multiple models are used for robust face detection:
- **MTCNN**: Handles multiple face angles and provides facial landmark detection.
- **RetinaFace**: Known for detecting faces in difficult conditions such as low light.
- **SCRFD** (InsightFace): Lightweight and fast, suitable for real-time detection.

*Combination rationale*: Combining models enhances precision in detecting faces under various conditions, such as occlusion or poor lighting, while maintaining fast performance.

### Face Embeddings
Face embeddings are generated using multiple models for higher accuracy:
- **ArcFace**: High accuracy, especially for large datasets.
- **FaceNet**: Generates compact embeddings with high recall.
- **VGG-Face**: Provides highly separable embeddings that improve overall accuracy.
- **InceptionResnetV1**: Suitable for both frontal and side views of faces.

*Combination rationale*: The variety of models ensures that the system can handle a wide range of conditions and face angles, improving the robustness of face recognition.

### Clustering and Matching
Multiple clustering methods are used to group similar faces across videos:
- **DBSCAN**: Filters noise and false positives.
- **KMeans**: Efficient for handling large datasets.
- **Spectral Clustering**: Suitable for complex data structures.
- **Agglomerative Clustering**: Hierarchical clustering for better structure in datasets.

*Combination rationale*: Using different clustering algorithms improves the precision of matching faces across multiple videos, even in noisy datasets.

### Metadata and Logging
- **Folder-based structure**: Each detected face has a dedicated folder storing all occurrences of that face.
- **Metadata logs**: A log file in each folder tracks where and when the face appears across videos.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- DeepFace
- InsightFace
- Scikit-learn
- Facenet-PyTorch
- Torch

To install the required packages:
```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fersenhaIter/cross_video_identification.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place video files in the designated folder.
2. Run the main script:
   ```bash
   python main.py
   ```
3. The terminal interface allows custom folder names. If folders don’t exist, they will be created.
4. Detected faces are saved in individual folders, and metadata logs track when and where the face appears in the videos.

## Current Issues and Next Steps

1. **EXIF Data Extraction**: The system currently relies on `exiftool`, which can cause compatibility issues. Consider using **exifread** or **Pillow** for native Python EXIF handling.
2. **Hyperparameter Tuning**: Fine-tuning distance thresholds in clustering algorithms will improve face matching accuracy.
3. **Performance Optimization**: Implement GPU acceleration with CUDA to speed up embedding generation and clustering, especially for larger datasets.
4. **Enhanced User Interface**: The terminal UI can be improved using libraries like **Rich** or **BeautifulTerminal** to enhance user experience and provide better visualization.
5. **Real-time Processing**: Adding support for real-time processing with SCRFD can improve the system’s utility in live settings.

## Future Enhancements

- **GPU Acceleration**: Implement CUDA for real-time processing.
- **Web Interface**: Add a web-based UI for easier control and log viewing.
- **Cloud Integration**: Enable cloud-based storage and processing for large datasets.
