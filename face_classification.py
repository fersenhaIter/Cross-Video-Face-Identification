import cv2
from keras_facenet import FaceNet
import os
import sklearn.cluster as cl
from sklearn.metrics import silhouette_score

class FaceClassification():

    def __init__(self):
        self.facenet = FaceNet()
        self.embedding_data = {}

    def get_embeddings(self):
        count = 0
        for video_name in os.listdir("result"):
            if os.path.isdir("result/" + video_name):
                for face_img in os.listdir("result/" + video_name):
                    try:
                        img_data = cv2.imread("result/" + video_name + "/" + face_img)
                        img_data = cv2.resize(img_data, (160,160))
                        self.embedding_data[count] = {"img_name": face_img,
                                                      "video_source": video_name,
                                                      "img_data": img_data,
                                                      "path": "result/" + video_name + "/" + face_img}
                        count += 1
                    except cv2.error as e:
                        print("Invalid frame!")
        img_data_list = [self.embedding_data[img]["img_data"] for img in self.embedding_data]
        embeddings = self.facenet.embeddings(img_data_list)
        for embed in range(len(embeddings)):
            self.embedding_data[embed]["embedding"] = embeddings[embed]
        return self.embedding_data

    def get_classes(self):
        self.get_embeddings()
        X = [self.embedding_data[face]["embedding"] for face in self.embedding_data]
        # Erstelle das Dictionary der Hyperparameter-Werte
        eps_ = 0.6
        k = self.calculate_k(X, eps = eps_)
        labels = k["labels"]
        labels_to_img = {}
        for data_point in range(len(labels)):
            if not list(labels_to_img.keys()).__contains__(labels[data_point]):
                labels_to_img[labels[data_point]] = []
            labels_to_img[labels[data_point]].append(self.embedding_data[data_point])

        if not os.path.isdir("result/clustered" + "EPS" + str(eps_)):
            os.mkdir("result/clustered" + "EPS" + str(eps_))

        for label in labels_to_img:
            different_sources = False
            save_dir = "result/clustered" + "EPS" + str(eps_) + "/" + str(label)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            f =  open(save_dir+"/"+"info.txt", "x")
            f.close()
            f =  open(save_dir+"/"+"info.txt", "a")
            first_source = labels_to_img[label][0]["video_source"]
            for img in labels_to_img[label]:
                cv2.imwrite(save_dir + "/" + img["img_name"],img["img_data"])
                f.write(img["img_name"]+"       video : "+img["video_source"]+"     timestamp : "+img["img_name"].rsplit('.', 1)[0]+" sec\n")
                if img["video_source"] is not first_source:
                    different_sources = True
            f.close()
            if different_sources:
                f = open("result/multiple_sources.txt","w")
                f.close()
                f = open("result/multiple_sources.txt","a")
                f.write("face label nr.:"+str(label)+"\n")
                f.close()
    def calculate_k(self, X, eps):
        max_k = {"score": -1}
        # Berechne den Silhouetten-Score für k-Werte zwischen 2 und 10
        for k in range(2, len(self.embedding_data)):
            try:
                cluster = cl.DBSCAN(eps=eps)
                cluster.fit(X)
                labels = cluster.fit_predict(X)

                score = silhouette_score(X, cluster.labels_)
                if score >= max_k["score"]:
                    max_k = {"k": k, "score": score, "labels": labels}
            except ValueError:
                pass
        return max_k