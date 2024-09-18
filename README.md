# Video Face Detection for Cross-Video Person Identification

<img src="cover_.jpg" alt="Project Cover" align="right" style="width: 200px; border: 2px solid black; margin-left: 20px; float: right;"/>

This project processes video footage to detect faces, generate embeddings, and attempt to identify recurring individuals across multiple videos. Originally designed for war journalism, particularly during evacuations and rescue missions, the system helps identify the same individuals in different videos. Through recent optimizations, frame selection and face recognition performance have been improved, and a combination of embedding models and clustering techniques are now used to increase accuracy.

## Features

- **Optimized Frame Selection**: The system intelligently selects frames to avoid redundant processing based on motion detection (optical flow) and brightness analysis, improving efficiency while preserving critical data.
- **Face Detection**: Extracts faces from video frames using models such as MTCNN and RetinaFace.
- **Combined Face Embeddings**: Generates face embeddings using multiple models (e.g., VGG-Face, ArcFace, OpenFace, FaceNet) to create a robust high-dimensional representation of faces.
- **Cross-video Matching with Hybrid Clustering**: Utilizes a combination of clustering algorithms like Spectral Clustering, DBSCAN, and Gaussian Mixture Models (GMM) to group and match similar faces across different videos.
- **Metadata Storage**: Detected face images are saved in a structured directory, with corresponding metadata (video source, timestamp, image path) logged in a JSON file.

## Frame Selection Optimization

To increase efficiency, the program implements two primary strategies for selecting frames:

1. **Motion-Based Frame Skipping**: Uses optical flow to detect significant motion between frames. Similar consecutive frames are skipped.
2. **Brightness-Based Selection**: Analyzes brightness changes across frames, skipping those with little variation to avoid redundant processing in dimly lit scenes.

These techniques help reduce the processing of unnecessary or low-quality frames, optimizing overall performance while retaining essential data for face detection.

## Face Embeddings and Clustering

### Face Embeddings

Face embeddings are vector representations of faces in a high-dimensional space. The system uses multiple deep learning models to convert each detected face into an embedding, where similar faces will have closer embeddings in this space. To ensure robust recognition, several pre-trained models are used:

- **VGG-Face**
- **OpenFace**
- **ArcFace**
- **FaceNet**

The embeddings from these models are normalized and combined to leverage their individual strengths, producing a more accurate representation of each face.

### Clustering Algorithms

To match faces across different videos, the system clusters embeddings based on their proximity in the feature space. A combination of clustering methods improves the system’s accuracy in identifying recurring faces:

1. **Spectral Clustering**: A graph-based algorithm that partitions data into clusters by analyzing eigenvalues of similarity matrices.
2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: A density-based algorithm that groups closely packed points and labels outliers as noise, ideal for filtering false positives.
3. **Gaussian Mixture Model (GMM)**: A probabilistic model that assumes data points are generated from multiple Gaussian distributions, refining clusters and improving identification accuracy.

By combining these algorithms, the system can accurately identify clusters of recurring faces across multiple videos, even in noisy or complex data.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- DeepFace
- Scikit-learn
- Keras (for VGG-Face preprocessing)
- Facenet-PyTorch
- Torch

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fersenhaIter/cam_analysis.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your video files in a designated directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Detected faces will be saved in the `result/` directory, and metadata is stored in `data.json`.

## Program Structure

- `main.py`: The core script that orchestrates the entire process.
- `face_detect.py`: Detects faces from video frames using models like MTCNN and RetinaFace.
- `face_classification.py`: Generates face embeddings using multiple models and clusters them to match individuals across videos.
- `data.json`: Logs metadata for each detected face, including the image name, video source, and file path.

## Future Improvements

- Further optimize cross-video matching accuracy.
- Introduce real-time video processing capabilities.
- Leverage GPU acceleration for faster analysis.
- Implement enhanced face tracking methods for better consistency.
