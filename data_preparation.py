import cv2
import mtcnn
from retinaface import RetinaFace
from time import time


class dataPreparation():

    def __init__(self):

        self.cascade_face_detector = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
        self.hog_face_detector = cv2.HOGDescriptor()
        self.hog_face_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.mtcnn_face_detector = mtcnn.MTCNN()
        
        self.count = 0

    def overlapping_area(self, p0 ,p1 ,p2 ,p3):
        x, y = 0,1

        x_left = max(p0[x], p2[x])
        y_top = max(p0[y], p2[y])
        x_right = min(p1[x], p3[x])
        y_bottom = min(p1[y], p3[y])

        area = 0
        if x_left <= x_right and y_top <= y_bottom:
            overlap_area = (x_right-x_left)*(y_bottom-y_top)

            x_left = min(p0[x], p2[x])
            y_top = min(p0[y], p2[y])
            x_right = max(p1[x], p3[x])
            y_bottom = max(p1[y], p3[y])

            max_rect_area = (x_right-x_left)*(y_bottom-y_top)
            relative_overlapping = overlap_area/max_rect_area

            if(relative_overlapping>0.5): return (x_left,y_top),(x_right,y_bottom)
            return None
        return None


    def get_faces(self,frame):

        frame_faces = {}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mtcnn_faces = self.mtcnn_face_detector.detect_faces(rgb_frame)

        retina_faces = RetinaFace.detect_faces(frame)

        if not isinstance(retina_faces, dict):
            return frame_faces

        #remove every entry with faces smaller than 95px
        #retina_faces = {k: v for k, v in retina_faces.items() if v.get("facial_area")[2]-v.get("facial_area")[0]>=95 and v.get("facial_area")[3]-v.get("facial_area")[1]>=95}
        #mtcnn_faces = [d for d in mtcnn_faces if d.get("box")[2] and d.get("box")[3]>=95]

        if (len(retina_faces) * len(mtcnn_faces) != 0):
            for face in mtcnn_faces:
                #beginning- and endingpoints
                x, y, w, h = face["box"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                for face in retina_faces.keys():
                    #beginning- and endingpoints
                    bx, by, ex, ey = retina_faces[face]["facial_area"]
                    cv2.rectangle(frame, (ex, ey), (bx, by), (255, 0, 0), 1)
                    rectangle = self.overlapping_area((x,y),(x+w,y+h),(bx,by),(ex,ey))
                    if(not isinstance(rectangle,tuple)):
                        break
                    (p1,p2),(p3,p4) = self.overlapping_area((x,y),(x+w,y+h),(bx,by),(ex,ey))
                    face_name = str(round(self.count/(29.97),3))
                    while(frame_faces.keys().__contains__(face_name)):
                        face_name+="Q"
                    frame_faces[face_name] = ((p1,p2),(p3,p4))
        return frame_faces

    def store_image(self, frame, frame_faces):
        for face in frame_faces:
            (p1, p2), (p3, p4) = frame_faces[face]
            # Region of Interest
            roi = frame[p2:p4, p1:p3]
            cv2.rectangle(frame, (p1, p2), (p3, p4), (255, 255, 255), 1)
            print(face + ".jpg")
            cv2.imwrite("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/faces/" + face + ".jpg", frame)

    def get_th_frames(self, video_string):
        self.vid = cv2.VideoCapture(video_string)

        while self.vid.isOpened():
            ret, frame = self.vid.read()
            self.count += 1
            if(ret and self.count%5 == 0):
                print(self.count/29.97)
                self.store_image(frame, self.get_faces(frame))

        # After the loop release the cap object
        self.vid.release()

dataPreparation1 = dataPreparation()
dataPreparation1.get_th_frames("C:/Users/jakob/Pictures/Camera Roll/WIN_20230401_21_37_45_Pro.mp4")
#dataPreparation1.get_th_frames("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/4jakob/GX010321.mp4")
