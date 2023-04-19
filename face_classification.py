import cv2
import numpy as np
import openface
from deepface import DeepFace
from keras_facenet import FaceNet
import os
import sklearn.cluster as cl
from numpy import asarray, expand_dims
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

model = VGGFace(model='senet50', include_top=True, input_shape=(224, 224, 3), pooling='max')
model1 = DeepFace.build_model('Facenet')
# TODO: combine VGGFace and FACEnet

class FaceClassification():

    def __init__(self):
        self.facenet = FaceNet()
        self.embedding_data = {}

    # extract faces and calculate face embeddings for a list of photo files
    def get_vgg_embedding(self, img):
        # extract faces
        img = cv2.resize(img,(224,224))
        img = asarray(img, 'float32')
        img = expand_dims(img, axis=0)
        # prepare the face for the model, e.g. center pixels

        img = preprocess_input(img, version=2)
        # perform prediction
        embedding = model.predict(img, verbose=False)
        return embedding

    def get_deepface_embedding(self, img):
        img = cv2.resize(img,(160,160))
        img = asarray(img, 'float32')
        img = expand_dims(img, axis=0)
        embedding = model1.predict(img,verbose=False)
        return embedding

    def get_embeddings(self):
        count = 0
        for video_name in os.listdir("result"):
            if os.path.isdir("result/" + video_name):
                for face_img in os.listdir("result/" + video_name):
                    try:
                        img_data = cv2.imread("result/" + video_name + "/" + face_img)
                        img_path = "result/" + video_name + "/" + face_img
                        embedding1 = self.get_vgg_embedding(img_data)
                        embedding2 = self.get_deepface_embedding(img_data)
                        if embedding1 is not None and embedding2 is not None:
                            self.embedding_data[count] = {"img_name": face_img,
                                                          "video_source": video_name,
                                                          "img_data": cv2.resize(img_data, (160,160)),
                                                          "path": img_path,
                                                          "embedding":np.concatenate((np.array(embedding1).flatten(),np.array(embedding2).flatten()))}
                            count += 1
                    except cv2.error:
                        print("Invalid frame!")
        img_data_list = [self.embedding_data[img]["img_data"] for img in self.embedding_data]
        embeddings = self.facenet.embeddings(img_data_list)
        for embed in range(len(embeddings)):
            self.embedding_data[embed]["embedding"] = np.concatenate((np.array(embeddings[embed]), np.array(self.embedding_data[embed]["embedding"]).flatten()))
        return self.embedding_data

    def get_classes(self):
        self.get_embeddings()
        X = np.array([self.embedding_data[face]["embedding"] for face in self.embedding_data])
        X = X.reshape(X.shape[0],-1)
        #X = StandardScaler().fit_transform(X)
        # Erstelle das Dictionary der Hyperparameter-Werte
        '''for eps_ in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.72,0.74,0.76,0.78,0.8,0.85,0.9]:
        labels = self.calculate_k(X, eps = eps_)'''
        label_list = self.calculate_labels(X)
        for eps_ in label_list:
            labels = label_list[eps_]
            labels_to_img = {}
            for unique_label in np.unique(labels):
                labels_to_img[str(unique_label)] = []
            for data_point in range(len(labels)):
                labels_to_img[str(labels[data_point])].append(self.embedding_data[data_point])

            out_saving_path = "result/clustered_" + "EPS_" + str(round(eps_,3))
            if not os.path.isdir(out_saving_path):
                os.mkdir(out_saving_path)

            for label in labels_to_img:
                different_sources = False
                save_dir = out_saving_path + "/" + str(label) +"/"
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                for face in labels_to_img[label]:
                    cv2.imwrite(save_dir+"/"+face["img_name"],face["img_data"])
                f = open(save_dir + "info.txt", "x")
                f.close()
                f = open(save_dir + "info.txt", "a")
                first_source = labels_to_img[label][0]["video_source"]
                for img in labels_to_img[label]:
                    cv2.imwrite(save_dir + "/" + img["img_name"], img["img_data"])
                    f.write(img["img_name"] + "       video : " + img["video_source"] + "     timestamp : " +
                            img["img_name"].rsplit('.', 1)[0] + " sec\n")
                    if img["video_source"] is not first_source:
                        different_sources = True
                f.close()
    def calculate_k(self, X):
        max_k = {"labels":cl.KMeans(n_clusters=2).fit_predict(), "score":None}
        for k in range(2, len(X)-1):
            kmeans = cl.KMeans(n_clusters=k)
            clusters = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, clusters)

            # Calinski-Harabasz Score berechnen
            ch_score = calinski_harabasz_score(X, clusters)
            score = silhouette_avg*ch_score
            if max_k["score"] is None or max_k["score"] < score:
                max_k = {"labels":clusters, "score":score}
        return max_k

    def calculate_labels(self,X):
        max_k = {}
        # Berechne den Silhouetten-Score für k-Werte zwischen 2 und 10
        eps = 0.1
        while len(np.unique(cl.DBSCAN(eps=eps).fit_predict(X))) == 1:
            eps+=0.1
            print(eps, len(np.unique(cl.DBSCAN(eps=eps).fit_predict(X))))
        unique = len(np.unique(cl.DBSCAN(eps=eps).fit_predict(X)))
        while 1 < unique < len(X):
            dbscan = cl.DBSCAN(eps=eps)
            clusters = dbscan.fit_predict(X)

            silhouette_avg = silhouette_score(X, clusters)

            # Calinski-Harabasz Score berechnen
            ch_score = calinski_harabasz_score(X, clusters)

            score = silhouette_avg * ch_score

            print(eps, score, unique, clusters)

            max_k[eps] = clusters
            eps += 0.1
            unique = len(np.unique(cl.DBSCAN(eps=eps).fit_predict(X)))
        return max_k