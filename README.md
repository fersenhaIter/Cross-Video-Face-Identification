# Cross-Video Face Identification

<img src="cover_.jpg" alt="Project Cover" align="right" style="width: 200px; border: 2px solid black; margin-left: 20px; float: right;"/>

## Overview

This project processes video footage to detect faces, generate embeddings, and identify recurring individuals across multiple videos. Originally designed for war journalism and evacuations, it has been optimized for accuracy and performance. The system allows users to customize video directories, creating individual folders for each detected face, along with logs that track where each face appears across different videos. By using multiple face detection and embedding models, the system ensures higher precision and performance.

## Key Features

### Optimized Frame Selection
The system applies advanced techniques to avoid redundant frame processing, boosting both accuracy and efficiency:
1. **Motion Detection**: Frames are skipped if minimal motion is detected using **optical flow** techniques. This ensures that the system only processes frames with significant motion changes, reducing unnecessary computations.
2. **Brightness Analysis**: Low-light frames are filtered out by comparing brightness histograms. This ensures that the system avoids processing frames that are too dark to provide useful data.

*Reason for combination*: Combining motion detection and brightness analysis helps the system filter out both redundant frames and low-quality frames, significantly reducing the time spent processing frames that do not contribute meaningful data.

### Face Detection
Multiple face detection models are used in tandem to increase precision, making the system robust to variations in lighting, face angles, and occlusions:
- **MTCNN** (Multi-task Cascaded Convolutional Networks): Effective for detecting facial landmarks and recognizing faces from multiple angles.
- **RetinaFace**: Robust for detecting faces in low-light conditions and complex environments.
- **SCRFD** (InsightFace): Lightweight and optimized for real-time detection, providing a balance of speed and accuracy.

*Reason for combination*: By combining **MTCNN**'s landmark detection with **SCRFD**’s real-time performance, the system ensures precise face detection even in challenging environments while remaining computationally efficient.

### Face Embeddings
Face embeddings are generated using multiple state-of-the-art models to ensure robust face recognition:
- **ArcFace** (InsightFace): Provides highly accurate embeddings, especially for large datasets.
- **FaceNet**: Known for generating compact embeddings with high recall, useful for diverse face datasets.
- **VGG-Face**: Produces highly separable embeddings, which are especially useful when combined with other models.
- **InceptionResnetV1**: Reliable for generating embeddings for both frontal and side faces.

*Reason for combination*: Using multiple models ensures that the system produces robust face embeddings across different scenarios. By normalizing and combining embeddings from multiple models, the system enhances accuracy and reduces the risk of false matches.

### Clustering and Matching
To identify recurring faces across multiple videos, the project uses a hybrid clustering approach:
- **DBSCAN**: Density-based clustering for filtering out noise and false positives, useful in large datasets with variable densities.
- **KMeans**: A fast clustering algorithm that handles large datasets with predefined cluster numbers.
- **Spectral Clustering**: Excellent for complex, graph-based partitioning of data.
- **Agglomerative Clustering**: A hierarchical approach, ideal for nested or hierarchical relationships between faces.

*Reason for combination*: Each clustering algorithm has unique strengths. **DBSCAN** excels at filtering noise, while **KMeans** is useful for handling large datasets quickly. **Agglomerative Clustering** is suitable for creating hierarchical structures when faces appear in different contexts. Combining these methods ensures precise clustering of recurring faces across multiple videos.

### Metadata and Logging
- **Folder-based structure**: A separate folder is created for each detected face, storing all instances of that face across different videos.
- **Metadata logs**: Each folder contains a log file that tracks where and when the face was detected in each video, aiding in the analysis of recurring individuals.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- DeepFace
- InsightFace
- Scikit-learn
- Facenet-PyTorch
- Torch

To install the required packages, run:
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

1. Place your video files in the designated folder.
2. Run the main script:
   ```bash
   python main.py
   ```
3. The terminal interface allows you to input custom folder names. If folders don’t exist, the system will create them.
4. Detected faces are saved in individual folders, and metadata logs are created for each face, containing information about where and when the face appears in different videos.

## Current Issues and Next Steps

1. **EXIF Data Extraction**: The current program relies on `exiftool`, but there have been compatibility issues. Consider replacing it with a Python-native library like **exifread** or **Pillow** for metadata extraction.
2. **Face Matching Accuracy**: The combination of models is effective, but further tuning of hyperparameters, such as distance thresholds in clustering, will enhance accuracy.
3. **Performance Optimization**: GPU acceleration (CUDA) should be implemented to speed up face embedding generation and clustering, especially for large datasets.
4. **Improved User Interface**: The terminal UI could be enhanced using frameworks like **Rich** or **BeautifulTerminal** to provide better visualization and debugging options.
5. **Real-time Processing**: Adding real-time processing support will allow for live face identification using models like SCRFD for faster results.

## Future Enhancements

- **GPU Acceleration**: Implementing CUDA for large-scale video processing and real-time face detection.
- **Web-based Interface**: Introducing a web-based user interface to make input and log viewing more user-friendly.
- **Cloud Integration**: Support for cloud-based storage and distributed processing to handle larger datasets more efficiently.
