import cv2
import numpy as np
import json
import torch
from deepface import DeepFace
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os
from numpy import asarray, expand_dims
from keras_vggface.utils import preprocess_input
from facenet_pytorch import InceptionResnetV1
from keras_facenet import FaceNet

facenet_vggface2_model = InceptionResnetV1(pretrained='vggface2').eval()
models = [[DeepFace.build_model('DeepFace'),(152, 152)],
          [DeepFace.build_model('OpenFace'),(96, 96)],
          [DeepFace.build_model('VGG-Face'),(224, 224)],
          [DeepFace.build_model('ArcFace'),(112, 112)]]

class FaceClassification():

    def __init__(self):
        self.facenet = FaceNet()
        self.embedding_data = {}


    def preprocess_img(self,img, shape):
        img = cv2.resize(img,shape)
        img = asarray(img, 'float32')
        img = expand_dims(img, axis=0)
        img = preprocess_input(img, version=2)
        return img
    # extract faces and calculate face embeddings for a list of photo files
    def get_some_embeddings(self, img):
        face = cv2.resize(img, (224, 224))
        face = np.expand_dims(face, axis=0)
        face = face.astype('float32')
        face /= 255.0
        face_net_embedding = facenet_vggface2_model(torch.from_numpy(face.transpose((0, 3, 1, 2)))).detach().numpy()
        face_net_embedding = face_net_embedding / np.linalg.norm(face_net_embedding.flatten(), axis=0,keepdims=True)

        multiple_way_embed = face_net_embedding

        for model_no in range(len(models)):
            model, shape = models[model_no]
            embedding = model.predict(self.preprocess_img(img, shape), verbose=False)
            embedding = embedding / np.linalg.norm(np.array(embedding).flatten(), axis=0,keepdims=True)
            multiple_way_embed = np.concatenate((multiple_way_embed,embedding), axis=1)
        return multiple_way_embed

    def get_embeddings(self):
        if os.path.exists("data.json"):
            print("load data...")
            stored_data = json.load(open("data.json", "r"))
            for face in stored_data:
                stored_data[face]["embedding"] = normalize(np.load("result/embeddings/"+str(face)+".npy").reshape(-1, 1),axis=0)
            self.embedding_data = stored_data
            print("data loaded!")

        else:
            if not os.path.isdir("result/embeddings"):
                os.mkdir("result/embeddings")
            print("calculating embedding data...")
            temp_embedding_data = {}
            count = 0
            for folder_name in os.listdir("result"):
                if os.path.isdir("result/" + folder_name) and folder_name != "embeddings":
                    video_name = folder_name
                    for face_img in os.listdir("result/" + video_name):
                        img_path = "result/" + video_name + "/" + face_img
                        img_data = cv2.imread(img_path)
                        if img_data.size == 0:
                            break
                        img_path = "result/" + video_name + "/" + face_img
                        embedding = self.get_some_embeddings(img_data)
                        embedding = np.array(embedding).flatten() / np.linalg.norm(np.array(embedding).flatten(), axis=0, keepdims=True)

                        self.embedding_data[str(count)] = {"img_name": face_img,
                                                      "video_source": video_name,
                                                      "path": img_path,
                                                      "embedding":embedding}
                        temp_embedding_data[str(count)] = {"img_name": face_img,
                                                           "video_source": video_name,
                                                           "path": img_path}
                        count += 1
            json.dump(temp_embedding_data, open("data.json", "w"))
            img_data_list = [cv2.resize(cv2.imread(self.embedding_data[img]["path"]), (160,160)) for img in self.embedding_data]
            embeddings = self.facenet.embeddings(img_data_list)
            for embed in range(len(embeddings)):
                embedding1 = np.array(embeddings[embed])
                embedding1 = embedding1/ np.linalg.norm(embedding1, axis=0, keepdims=True)
                embedding = normalize(np.concatenate((embedding1, self.embedding_data[str(embed)]["embedding"])).reshape(-1,1),axis=0)
                self.embedding_data[str(embed)]["embedding"] = embedding
                np.save("result/embeddings/"+str(embed)+".npy", embedding)
            print("embedding data calculated and stored!")

        return self.embedding_data

    def all_label(self,X):
        max_k = {}
        n_list = range(18,23)
        gamma = [.4,.5,.6,.75,1,1.25,1.5,2]
        eps_ = [.5,.6,.65,.67,.69,.7,.72,.8,.9,1,1.1,1.2]
        for g in eps_:
            #for n in n_list:
            #clust = SpectralClustering(n_clusters=n, affinity='rbf', assign_labels='kmeans', eigen_solver="lobpcg", gamma=g, n_init=200)
            clust = DBSCAN(eps=g)
            clusters = clust.fit_predict(X)
            unique_len = len(np.unique(clusters))

            if not (unique_len < 2 or unique_len > len(X)-1):
                #scoring
                silhouette_avg = silhouette_score(X, clusters)
                ch_score = calinski_harabasz_score(X, clusters)
                score = silhouette_avg * ch_score

                print(g,silhouette_avg,ch_score,score,unique_len)
                max_k[str(g)] = clusters
        return max_k

    def st(self):
        self.get_embeddings()
        X = np.array([self.embedding_data[face]["embedding"] for face in self.embedding_data])
        X = X.reshape(X.shape[0], -1)
        # X = StandardScaler().fit_transform(X)
        # Erstelle das Dictionary der Hyperparameter-Werte
        label_dict = self.all_label(X)
        out_saving_path = "result/labeled/"
        if not os.path.isdir(out_saving_path):
            os.mkdir(out_saving_path)
        for eps in label_dict:
            labels = label_dict[eps]
            labels_to_img = {}
            for unique_label in np.unique(labels):
                labels_to_img[str(unique_label)] = []
            for data_point in range(len(labels)):
                labels_to_img[str(labels[data_point])].append(self.embedding_data[str(data_point)])

            save_dir = out_saving_path + eps
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            for label in labels_to_img:
                save_dir = out_saving_path+ eps +"/" + str(label) + "/"
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                if not os.path.exists(save_dir + "info.txt"):
                    f = open(save_dir + "info.txt", "x")
                    f.close()
                f = open(save_dir + "info.txt", "a")
                first_source = labels_to_img[label][0]["video_source"]

                for face in labels_to_img[label]:
                    cv2.imwrite(save_dir + face["img_name"], cv2.imread(face["path"]))
                    f.write(face["img_name"] + "       video : " + face["video_source"] + "     timestamp : " +
                            face["img_name"].rsplit('.', 1)[0] + " sec\n")
                f.close()


    def get_classes(self):
        self.get_embeddings()
        X = np.array([self.embedding_data[face]["embedding"] for face in self.embedding_data])
        X = X.reshape(X.shape[0],-1)
        #X = StandardScaler().fit_transform(X)
        # Erstelle das Dictionary der Hyperparameter-Werte
        labels = self.all_label(X)
        labels_to_img = {}
        for unique_label in np.unique(labels):
            labels_to_img[str(unique_label)] = []
        for data_point in range(len(labels)):
            labels_to_img[str(labels[data_point])].append(self.embedding_data[str(data_point)])

        out_saving_path = "result/labeled/"
        if not os.path.isdir(out_saving_path):
            os.mkdir(out_saving_path)

        for label in labels_to_img:
            different_sources = False
            save_dir = out_saving_path + str(label) +"/"
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            if not os.path.exists(save_dir + "info.txt"):
                f = open(save_dir + "info.txt", "x")
                f.close()
            f = open(save_dir + "info.txt", "a")
            first_source = labels_to_img[label][0]["video_source"]

            for face in labels_to_img[label]:
                cv2.imwrite(save_dir+face["img_name"],cv2.imread(face["path"]))
                f.write(face["img_name"] + "       video : " + face["video_source"] + "     timestamp : " +
                        face["img_name"].rsplit('.', 1)[0] + " sec\n")
                if face["video_source"] is not first_source:
                    different_sources = True
            f.close()
            if different_sources:
                f = open("result/multiple_sources.txt", "w")
                f.close()
                f = open("result/multiple_sources.txt", "a")
                f.write("face label nr.:" + str(label) + "\n")
                f.close()
