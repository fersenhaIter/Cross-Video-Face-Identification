import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
from pathlib import Path
from colorama import Fore, Style

# Import advanced face detection methods
import insightface
from insightface.app import FaceAnalysis
from retinaface import RetinaFace  # RetinaFace detector
from facenet_pytorch import MTCNN  # MTCNN detector

class FaceDetection:
    def __init__(self, logger):
        self.logger = logger
        self.setup_face_detectors()

    def setup_face_detectors(self):
        try:
            # Initialize multiple face detectors
            self.face_detectors = {
                'insightface': FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']),
                'retinaface': RetinaFace,
                'mtcnn': MTCNN(keep_all=True, device='cpu')
            }
            self.face_detectors['insightface'].prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info(Fore.GREEN + "Face detectors initialized successfully.")
        except Exception as e:
            self.logger.error(Fore.RED + f"Error setting up face detectors: {str(e)}")
            raise

    def process_videos(self, input_dir):
        video_paths = [os.path.join(root, file) for root, _, files in os.walk(input_dir)
                       for file in files if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        if not video_paths:
            self.logger.warning(Fore.YELLOW + f"No video files found in {input_dir}")
            return

        os.makedirs("result", exist_ok=True)
        self.logger.info(Fore.BLUE + "Starting video processing...\n")

        for video_path in tqdm(video_paths, desc=Fore.CYAN + "Processing videos", unit="video"):
            try:
                self.process_video(video_path)
                self.logger.info(Fore.GREEN + f"Processed video: {os.path.basename(video_path)}")
            except Exception as e:
                self.logger.error(Fore.RED + f"Error processing video {video_path}: {str(e)}")

    def process_video(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                self.logger.error(Fore.RED + f"Could not open video file: {video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            video_name = os.path.basename(video_path).rsplit('.', 1)[0]
            os.makedirs(f"result/{video_name}", exist_ok=True)

            frame_indices = self.select_frames(cap, total_frames, fps)

            for frame_number in tqdm(frame_indices, desc=f"Processing frames of {video_name}", unit="frame"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    self.logger.error(Fore.RED + f"Failed to read frame {frame_number} from {video_path}")
                    continue

                faces = self.detect_faces(frame)
                num_faces = len(faces)
                self.logger.debug(Fore.BLUE + f"Detected {num_faces} faces in frame {frame_number}")

                for idx, face_info in enumerate(faces):
                    aligned_face = face_info['aligned_face']
                    timestamp = frame_number / fps
                    face_filename = f"{timestamp:.2f}_{idx}.jpg"
                    cv2.imwrite(f"result/{video_name}/{face_filename}", aligned_face)

            cap.release()
        except Exception as e:
            self.logger.error(Fore.RED + f"Error processing video {video_path}: {str(e)}")

    def select_frames(self, cap, total_frames, fps):
        # Use advanced frame selection methods (e.g., scene detection)
        # For simplicity, select one frame per second
        frame_step = max(int(fps), 1)
        frame_indices = list(range(0, total_frames, frame_step))
        return frame_indices

    def detect_faces(self, frame):
        # Combine detections from multiple detectors
        faces = []

        # InsightFace detection
        insight_faces = self.face_detectors['insightface'].get(frame)
        for face in insight_faces:
            if self.is_high_quality_face(face, frame):
                aligned_face = face_align.norm_crop(frame, face.kps)
                faces.append({'aligned_face': aligned_face})

        # RetinaFace detection
        retina_faces = RetinaFace.detect_faces(frame)
        if isinstance(retina_faces, dict):
            for key in retina_faces.keys():
                face_data = retina_faces[key]
                facial_area = face_data['facial_area']
                x1, y1, x2, y2 = facial_area
                face_img = frame[y1:y2, x1:x2]
                if self.is_face_size_adequate(face_img):
                    faces.append({'aligned_face': face_img})

        # MTCNN detection
        boxes, _ = self.face_detectors['mtcnn'].detect(frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_img = frame[y1:y2, x1:x2]
                if self.is_face_size_adequate(face_img):
                    faces.append({'aligned_face': face_img})

        return faces

    def is_high_quality_face(self, face, frame):
        # Implement face quality assessment
        if not self.is_frontal_face(face):
            return False
        quality_score = self.assess_face_quality(face, frame)
        if quality_score < 0.4:
            return False
        return True

    def is_frontal_face(self, face):
        # Check the facial landmarks symmetry
        left_eye = face.kps[0]
        right_eye = face.kps[1]
        mouth_left = face.kps[3]
        mouth_right = face.kps[4]

        eye_distance = np.linalg.norm(left_eye - right_eye)
        mouth_distance = np.linalg.norm(mouth_left - mouth_right)
        ratio = eye_distance / mouth_distance if mouth_distance != 0 else 0

        if ratio < 0.5 or ratio > 2.0:
            return False
        return True

    def assess_face_quality(self, face, frame):
        # Face quality assessment based on sharpness and brightness
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        face_img = frame[y1:y2, x1:x2]

        if face_img.shape[0] < 50 or face_img.shape[1] < 50:
            return 0.0

        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        if laplacian_var < 30:
            return 0.0

        brightness = np.mean(gray_face)
        if brightness < 40 or brightness > 220:
            return 0.0

        sharpness_score = min(laplacian_var / 300, 1.0)
        brightness_score = 1.0 - abs(brightness - 128) / 128
        quality_score = (sharpness_score + brightness_score) / 2
        return quality_score

    def is_face_size_adequate(self, face_img):
        return face_img.shape[0] >= 50 and face_img.shape[1] >= 50
