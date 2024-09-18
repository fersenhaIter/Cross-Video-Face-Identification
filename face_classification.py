import os
import numpy as np
import json
import logging
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering, KMeans, OPTICS, Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from pathlib import Path
import cv2
from colorama import Fore, Style
from jinja2 import Environment, FileSystemLoader

# Import multiple embedding models
from deepface import DeepFace
from facenet_pytorch import InceptionResnetV1  # For FaceNet embeddings

class FaceClassification:
    def __init__(self, logger):
        self.logger = logger
        self.embedding_data = {}
        self.embeddings = []
        self.image_paths = []
        self.load_face_images()
        self.init_embedding_models()

    def load_face_images(self):
        result_dir = Path("result")
        for video_folder in result_dir.iterdir():
            if video_folder.is_dir():
                for face_image in video_folder.glob("*.jpg"):
                    self.image_paths.append(face_image)
        if not self.image_paths:
            self.logger.warning(Fore.YELLOW + "No face images found in 'result' directory.")
        else:
            self.logger.info(Fore.BLUE + f"Loaded {len(self.image_paths)} face images for embedding generation.")

    def init_embedding_models(self):
        # Initialize multiple embedding models
        self.embedding_models = {
            'ArcFace': DeepFace.build_model('ArcFace'),
            'Facenet512': DeepFace.build_model('Facenet512'),
            'VGG-Face': DeepFace.build_model('VGG-Face'),
            'InceptionResnetV1': InceptionResnetV1(pretrained='vggface2').eval()
        }

    def get_embeddings(self):
        if not self.image_paths:
            return
        self.embeddings = []
        self.logger.info(Fore.BLUE + "Generating embeddings...\n")
        for img_path in tqdm(self.image_paths, desc=Fore.CYAN + "Embedding images"):
            try:
                embedding = self.generate_embedding(str(img_path))
                self.embeddings.append(embedding)
                self.embedding_data[str(img_path)] = {
                    'embedding': embedding,
                    'image_path': str(img_path),
                    'video': img_path.parent.name,
                    'filename': img_path.name
                }
            except Exception as e:
                self.logger.error(Fore.RED + f"Error generating embedding for {img_path}: {str(e)}")

        self.embeddings = np.array(self.embeddings)
        np.save("embeddings.npy", self.embeddings)
        self.logger.info(Fore.GREEN + "Embeddings generated and saved.")

    def generate_embedding(self, img_path):
        embeddings = []
        img = DeepFace.functions.preprocess_face(img_path, target_size=(160, 160), enforce_detection=False)
        # Get embeddings from multiple models
        for model_name, model in self.embedding_models.items():
            if model_name == 'InceptionResnetV1':
                img_resized = cv2.resize(cv2.imread(img_path), (160, 160))
                img_normalized = (img_resized / 255.0).astype(np.float32)
                img_tensor = np.expand_dims(img_normalized.transpose(2, 0, 1), 0)
                embedding = model(torch.tensor(img_tensor)).detach().numpy().flatten()
            else:
                embedding = model.predict(img)[0]
            embeddings.extend(embedding)
        return np.array(embeddings)

    def cluster_faces(self, output_dir):
        if not self.embeddings.any():
            self.logger.warning(Fore.YELLOW + "No embeddings available for clustering.")
            return

        embeddings = StandardScaler().fit_transform(self.embeddings)
        self.logger.info(Fore.BLUE + "Clustering faces...\n")

        clustering_algorithms = [
            ('DBSCAN', DBSCAN(eps=0.6, min_samples=4, metric='euclidean', n_jobs=-1)),
            ('Agglomerative', AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)),
            ('Spectral', SpectralClustering(n_clusters=10, affinity='nearest_neighbors', n_jobs=-1)),
            ('KMeans', KMeans(n_clusters=10, n_init=10)),
            ('OPTICS', OPTICS(min_samples=5, n_jobs=-1)),
            ('Birch', Birch(n_clusters=None, threshold=0.5))
        ]

        best_score = -1
        best_labels = None
        best_algorithm = None
        for name, algorithm in clustering_algorithms:
            try:
                labels = algorithm.fit_predict(embeddings)
                if len(set(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    self.logger.info(Fore.GREEN + f"{name} clustering achieved silhouette score: {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_algorithm = name
                else:
                    self.logger.info(Fore.YELLOW + f"{name} clustering resulted in a single cluster.")
            except Exception as e:
                self.logger.error(Fore.RED + f"Error in {name} clustering: {str(e)}")

        if best_labels is None:
            self.logger.error(Fore.RED + "Clustering failed to produce valid labels.")
            return

        self.logger.info(Fore.BLUE + f"\nBest clustering algorithm: {best_algorithm} with silhouette score: {best_score:.4f}")
        self.save_clusters(best_labels, output_dir)

    def save_clusters(self, labels, output_dir):
        self.logger.info(Fore.BLUE + "\nSaving clustered faces...")
        self.clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue  # Skip noise
            person_dir = os.path.join(output_dir, f"person_{label}")
            os.makedirs(person_dir, exist_ok=True)

            img_path = self.image_paths[idx]
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(person_dir, img_name)
            cv2.imwrite(dest_path, cv2.imread(str(img_path)))

            # Save cluster data
            if label not in self.clusters:
                self.clusters[label] = {
                    'images': [],
                    'videos': set()
                }
            self.clusters[label]['images'].append({
                'filename': img_name,
                'path': dest_path,
                'video': img_path.parent.name
            })
            self.clusters[label]['videos'].add(img_path.parent.name)

        self.logger.info(Fore.GREEN + "Clustered faces saved successfully.")

    def generate_report(self, output_dir):
        self.logger.info(Fore.BLUE + "\nGenerating summary report...")
        # Prepare data for the report
        report_data = []
        for label, data in self.clusters.items():
            person_data = {
                'person_id': label,
                'num_images': len(data['images']),
                'num_videos': len(data['videos']),
                'videos': list(data['videos']),
                'sample_image': data['images'][0]['path'],
                'images': data['images']
            }
            report_data.append(person_data)

        # Generate HTML report using Jinja2
        env = Environment(loader=FileSystemLoader('.'))
        template = env.from_string(self.report_template())
        output_html = template.render(clusters=report_data)

        # Save the report
        report_path = os.path.join(output_dir, 'report.html')
        with open(report_path, 'w') as f:
            f.write(output_html)

        self.logger.info(Fore.GREEN + f"Summary report generated at {report_path}")

    def report_template(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Face Clustering Report</title>
            <style>
                body { font-family: Arial, sans-serif; }
                .person { margin-bottom: 40px; }
                .person h2 { color: #2e6c80; }
                .images { display: flex; flex-wrap: wrap; }
                .images img { margin: 5px; width: 100px; height: 100px; object-fit: cover; }
            </style>
        </head>
        <body>
            <h1>Face Clustering Report</h1>
            {% for person in clusters %}
            <div class="person">
                <h2>Person {{ person.person_id }}</h2>
                <p>Number of images: {{ person.num_images }}</p>
                <p>Appears in videos: {{ person.videos | join(', ') }}</p>
                <h3>Images:</h3>
                <div class="images">
                    {% for image in person.images %}
                    <img src="{{ image.path }}" alt="{{ image.filename }}">
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </body>
        </html>
        """
