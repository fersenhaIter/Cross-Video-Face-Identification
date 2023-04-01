import cv2
import mtcnn
from retinaface import RetinaFace
from time import time


class dataPreparation():

    def __init__(self):

        # video capture object
        self.vid = cv2.VideoCapture("C:\\Users\\jakob\\Downloads\\gkd_4jakob_2023-03-30_1342\\4jakob\\20220428_131159.mp4")

        self.cascade_face_detector = cv2.CascadeClassifier("models\\haarcascade_frontalface_default.xml")
        self.hog_face_detector = cv2.HOGDescriptor()
        self.hog_face_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.mtcnn_face_detector = mtcnn.MTCNN()
        
        self.count = 0

    def get_th_frames(self):

        while self.vid.isOpened():
            ret, frame = self.vid.read()
            self.count += 1
            if(ret and self.count%5 == 0):
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                cascade_faces = self.cascade_face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

                hog_faces, _ = self.hog_face_detector.detectMultiScale(gray_frame)

                mtcnn_faces = self.mtcnn_face_detector.detect_faces(rgb_frame)

                retina_faces = RetinaFace.detect_faces(frame)
                
                for (x, y, w, h) in cascade_faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                for hog_face in hog_faces:
                    (x, y, w, h) = hog_face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                for face in mtcnn_faces:
                    x, y, w, h = face["box"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                if isinstance(retina_faces , dict):
                    for face in retina_faces.keys():
                        recognize_face_area = retina_faces[face]["facial_area"]
                        score = round(retina_faces[face]["score"],3)
                        cv2.rectangle(frame, (recognize_face_area[2], recognize_face_area[3]), (recognize_face_area[0], recognize_face_area[1]), (255, 255, 255), 1)
                        cv2.putText(frame, str(score), (recognize_face_area[2], recognize_face_area[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                        
                # show frame
                cv2.imshow('Frame', frame)

            # 'q' as exit nessesary
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        self.vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

dataPreparation1 = dataPreparation()
dataPreparation1.get_th_frames()
