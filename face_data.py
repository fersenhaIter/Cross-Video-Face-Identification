import cv2
import face_detection
import sys, os
import torch
from mtcnn import MTCNN

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class FaceDetection():

    def __init__(self):
        self.mtcnn_face_detector = MTCNN()
        self.retina_net_res_net = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)
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

        mtcnn_faces =  self.mtcnn_face_detector.detect_faces(frame)
        if len(mtcnn_faces)==0:
            return frame_faces
        retina_net_res_net_faces = [(x1, y1, x2, y2, score) for (x1, y1, x2, y2, score) in self.retina_net_res_net.detect(frame).astype(int) if x2-x1>80 and y2-y1>80]
        if(len(retina_net_res_net_faces)==0):
            return frame_faces

        for face in retina_net_res_net_faces:
            x_1, y_1, x_2, y_2, score = face

            for face in mtcnn_faces:
                x,y,w,h = face["box"]
                rect = self.overlapping_area((x,y),(x+w,y+h),(x_1,y_1),(x_2,y_2))
                if (rect == None): break
                (x1,x2),(x3,x4) = rect
                face_name = str(round(self.count / (29.97), 3))
                while (frame_faces.keys().__contains__(face_name)):
                    face_name += "Q"
                frame_faces[face_name] = ((x1, x2), (x3, x4))
        return frame_faces

    def store_image(self, frame, frame_faces,path):
        for face in frame_faces:
            (p1, p2), (p3, p4) = frame_faces[face]
            # Region of Interest
            roi = frame[p2:p4, p1:p3]
            cv2.imwrite(path + str(self.count) + ".jpg", roi)

    def get_video_frame_faces(self, video_string, path):
        vid_faces = {}
        self.vid = cv2.VideoCapture(video_string)
        total_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_number in range(0, total_frames, 5):
            self.count += 5
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            _, frame = self.vid.read()
            frame_faces = self.get_faces(frame)
            if len(frame_faces) > 0:
                self.store_image(frame, frame_faces, path)
                vid_faces[self.count]=frame_faces
        # After the loop release the cap object
        self.vid.release()
        return vid_faces

dataPreparation1 = FaceDetection()
video_faces = dataPreparation1.get_video_frame_faces("C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/4jakob/20220804_103852.mp4", "C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/faces/")
#video_faces = dataPreparation1.get_th_frames("C:/Users/jakob/Pictures/Camera Roll/WIN_20230401_21_37_45_Pro.mp4","C:/Users/jakob/Downloads/gkd_4jakob_2023-03-30_1342/faces/")
for key in video_faces:
    print(key,video_faces[key])