import cv2
from keras_facenet import FaceNet
import numpy as np

class FaceClassification():

    def __init__(self):
        self.facenet = FaceNet()

    def preprocess_image(self ,image_path):
        img = cv2.imread(image_path)
        #skaliere auf 160x160
        img = cv2.resize(img, (160, 160))
        #normalisierung der pixelintensivität (zwischen 0 und 1)
        img = img.astype("float") / 255.0
        #bild in np-array (shape = (160, 160, 3))
        img = np.array(img, dtype=np.float32)
        #hinzufügen von dimension (shape = (1, 160, 160, 3)) img[0] = Anzahl der Bilder
        img = np.expand_dims(img, axis=0)
        return img

    def get_embeddings(self, image_path):
        embeddings = self.facenet.embeddings(self.preprocess_image(image_path))
        return embeddings