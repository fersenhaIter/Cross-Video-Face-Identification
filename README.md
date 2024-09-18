# Video Face Detection for Cross-Video Person Identification

<img src="cover_.jpg" alt="Project Cover" align="right" width="250" style="border: 2px solid black;"/>

This project processes video footage to detect faces, generate embeddings, and attempt to identify recurring individuals across multiple videos. Originally designed for war journalism, particularly during evacuations and rescue missions, the system helps identify the same individuals in different videos. The system is currently not fully operational and faces challenges with efficiency and performance.

## Features

- **Face detection**: The system extracts faces from video frames using OpenCV and DeepFace models.
- **Face embeddings**: Generates face embeddings using multiple models (e.g., VGG-Face, OpenFace, ArcFace) to represent faces in a high-dimensional feature space.
- **Cross-video matching**: Uses clustering algorithms like Spectral Clustering and DBSCAN to identify and group similar faces across different videos.
- **Metadata storage**: Detected face images are saved in an organized directory, and corresponding metadata (such as the video source, timestamp, and image path) is logged in a JSON file.

## How Embeddings and Clustering Work

### Face Embeddings

Embeddings are vector representations of faces in a high-dimensional space. Each face is converted into a numerical representation (embedding) by deep learning models, where similar faces will have closer embeddings in this space. For this project, multiple pre-trained models are used to generate embeddings for more robust face recognition:

- **VGG-Face**
- **OpenFace**
- **ArcFace**
- **DeepFace**

Each of these models processes the face images slightly differently, producing embeddings of varying dimensionality. These embeddings are normalized to create a consistent and comparable representation of each face. The combination of multiple models increases the chance of accurate matching across different lighting conditions, angles, and occlusions.

### Clustering Algorithms

To match faces across different videos, clustering algorithms group embeddings that are close together in the feature space. This project uses the following clustering methods:

1. **Spectral Clustering**:
   - A graph-based clustering technique that partitions the data into clusters by using eigenvalues of similarity matrices.
   - Works well for identifying clusters with complex structures.
   - In the context of this project, Spectral Clustering helps group similar faces by analyzing the relationships between face embeddings and creating clusters of individuals appearing across videos.

2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
   - A clustering algorithm that groups points closely packed together, marking points in sparse regions as noise.
   - Effective at handling outliers (e.g., incorrectly detected faces or noise).
   - DBSCAN helps filter out faces that appear rarely (potentially noise or false positives) and focuses on the frequent, recurring faces in different videos.

These clustering algorithms, when combined, allow the system to identify clusters of recurring faces across videos, even in the presence of noise or outliers. However, the current implementation still faces challenges in terms of speed and accuracy, particularly when processing large video datasets or handling complex scenes with many faces.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- DeepFace
- Scikit-learn
- Keras (for VGG-Face preprocessing)
- Facenet-PyTorch
- Torch

To install the required packages, use:
```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
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

### `main.py`

This is the core script that orchestrates the entire process:

- **Face Detection**: Calls the `FaceDetection` class from `face_detect.py` to process videos and detect faces frame by frame.
- **Cross-video Matching**: Uses the `FaceClassification` class from `face_classification.py` to extract embeddings from the detected faces and attempt to match recurring individuals across different videos.
- **Data Handling**: Organizes detected face images in directories and updates the metadata in the JSON file.

### `face_detect.py`

Handles the detection of faces from video frames:

- **Video Processing**: Processes video files frame by frame using OpenCV and identifies faces using pre-trained models.
- **Face Extraction**: Extracts faces and stores them as individual image files in a structured directory (`result/`).

### `face_classification.py`

Responsible for generating face embeddings and attempting cross-video person identification:

- **Preprocessing**: Prepares the face images for model input by resizing and normalizing them.
- **Multiple Embedding Models**: Uses a combination of models (VGG-Face, ArcFace, OpenFace, DeepFace) to generate face embeddings.
- **Clustering for Identification**: Clusters face embeddings using algorithms like DBSCAN and Spectral Clustering to identify the same individual across different videos.

### `data.json`

This file logs the metadata for each detected face, including:

- **img_name**: The name of the extracted image file.
- **video_source**: The source video from which the face was extracted.
- **path**: The directory path where the face image is stored.

Example entry:
```json
{
  "0": {
    "img_name": "0.62.jpg",
    "video_source": "evacuation_video.mp4",
    "path": "result/evacuation_video/0.62.jpg"
  }
}
```

## Future Improvements

- **Advanced Cross-video Matching**: Enhance the cross-video matching accuracy by improving clustering methods or integrating more sophisticated matching techniques such as Siamese networks or triplet loss models.
- **Improved Performance**: Optimize the program to handle large datasets and perform real-time processing by leveraging GPU-based acceleration.
- **Robust Face Tracking**: Integrate face tracking for better consistency across frames, ensuring that the same person is accurately followed throughout the video.
- **Real-time Video Processing**: Extend the system to support real-time video input, which would be useful in live-reporting scenarios.
