import cv2
import mtcnn
import face_detection
import logging
import numpy as np
from models.face_detection_master.face_detection import detector

class dataPreparation():

    def __init__(self):

        #self.dnn_face_detector = cv2.dnn.readNetFromCaffe("C:/Users/jakob/Documents/private projekte/cam_analysis/models/deploy.prototxt.txt", "C:/Users/jakob/Documents/private projekte/cam_analysis/models/res10_300x300_ssd_iter_140000.caffemodel")
        self.viola_jones_detector = cv2.CascadeClassifier("C:/Users/jakob/Documents/private projekte/cam_analysis/models/haarcascade_frontalface_alt.xml")
        #self.retina_face_detector = detector.RetinaFace(gpu_id=-1)
        #self.cascade_face_detector = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
        #self.hog_face_detector = cv2.HOGDescriptor()
        #self.hog_face_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.mtcnn_face_detector = mtcnn.MTCNN()
        #self.dsfd_detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
        #self.retina_net_res_net = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)
        self.retina_net_mobile_net = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)

        self.count = 0

    def overlapping_area(self, p0 ,p1 ,p2 ,p3):
        x, y = 0,1

        x_left = max(p0[x], p2[x])
        y_top = max(p0[y], p2[y])
        x_right = min(p1[x], p3[x])
        y_bottom = min(p1[y], p3[y])

        area = 0
        if x_left >= x_right and y_top >= y_bottom:
            return None

        overlap_area = (x_right-x_left)*(y_bottom-y_top)

        x_left = min(p0[x], p2[x])
        y_top = min(p0[y], p2[y])
        x_right = max(p1[x], p3[x])
        y_bottom = max(p1[y], p3[y])

        max_rect_area = (x_right-x_left)*(y_bottom-y_top)
        relative_overlapping = overlap_area/max_rect_area

        if(relative_overlapping>0.5): return (x_left,y_top),(x_right,y_bottom)

        return None


    def get_faces(self,frame):

        frame_faces = {}

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mtcnn_faces =  self.mtcnn_face_detector.detect_faces(frame)
        retina_net_mobile_net_faces = self.retina_net_mobile_net.detect(frame).astype(int)

        for face in retina_net_mobile_net_faces:
            x1, y1, x2, y2, score = face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            for face in mtcnn_faces:
                x,y,w,h = face["box"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                rect = self.overlapping_area((x,y),(x+w,y+h),(x1,y1),(x2,y2))
                if (rect == None): break
                (x1,x2),(x3,x4) = rect
                cv2.rectangle(frame, (x1, x2), (x3,x4), (255, 255, 255), 1)
                face_name = str(round(self.count / (29.97), 3))
                while (frame_faces.keys().__contains__(face_name)):
                    face_name += "Q"
                frame_faces[face_name] = ((x1, x2), (x3, x4))
        return frame_faces

    def store_image(self, frame, frame_faces):
        for face in frame_faces:
            (p1, p2), (p3, p4) = frame_faces[face]
            # Region of Interest
            roi = frame[p2:p4, p1:p3]
            #cv2.rectangle(frame, (p1, p2), (p3, p4), (255, 255, 255), 1)
            print(face + ".jpg")
            cv2.imwrite("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/faces/" + face + ".jpg", frame)

    def get_th_frames(self, video_string):
        self.vid = cv2.VideoCapture(video_string)

        while self.vid.isOpened():
            ret, frame = self.vid.read()
            self.count += 1
            if(ret and self.count%5 == 0):
                if len(self.get_faces(frame))>0:
                    cv2.imwrite(f"C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/faces/{self.count}.jpg", frame)
                '''cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break'''
        # After the loop release the cap object
        self.vid.release()

print(face_detection.available_detectors)

dataPreparation1 = dataPreparation()
#dataPreparation1.get_th_frames("C:/Users/jakob/Pictures/Camera Roll/WIN_20230401_21_37_45_Pro.mp4")
dataPreparation1.get_th_frames("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/4jakob/GX010321.mp4")
