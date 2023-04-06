import cv2
import face_detection
from datetime import datetime
import numpy
from mtcnn import MTCNN

class FaceDetection():

    def __init__(self):
        self.mtcnn_face_detector = MTCNN(min_face_size=90)
        self.retina_net_res_net = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)
        self.count = 0
        self.frame = numpy.ndarray

        # Enable CUDA for OpenCV
        cv2.cuda.setDevice(0)
        self.cuda_frame = cv2.cuda_GpuMat()

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

    def draw_facial_landmarks(self, keypoints):
        cv2.circle(self.frame, keypoints["left_eye"], radius=1, color=(0, 0, 255), thickness=1)
        cv2.circle(self.frame, keypoints["right_eye"], radius=1, color=(0, 0, 255), thickness=1)
        cv2.circle(self.frame, keypoints["nose"], radius=1, color=(0, 0, 255), thickness=-1)
        cv2.line(self.frame,keypoints["mouth_left"], keypoints["mouth_right"], color=(0, 0, 255), thickness=1)

    def get_faces(self):
        frame_faces = {}

        mtcnn_faces =  [detected_face for detected_face in self.mtcnn_face_detector.detect_faces(self.frame) if detected_face["confidence"]>= 0.9]
        mtcnn_faces =  [detected_face for detected_face in mtcnn_faces if detected_face["box"][2]>= 96 and detected_face["box"][3]>= 96]
        if (mtcnn_faces.__len__() == 0):
            return frame_faces

        # copy frame to GPU
        self.cuda_frame.upload(self.frame)
        # calc faces on GPU
        retina_faces = self.retina_net_res_net.detect(self.cuda_frame).astype(int)

        # copy results back to CPU memory
        retina_faces = retina_faces.get()
        for (x1, y1, x2, y2, score) in retina_faces:
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = max(x2, 0)
            y2 = max(y2, 0)
            if x2 - x1 > 96.0 and y2 - y1 > 96.0:
                face_name = str(round(self.count / 7.25, 2))
                while (frame_faces.keys().__contains__(face_name)):
                    face_name += "0"
                frame_faces[face_name] = ((int(x1), int(y1)), (int(x2), int(y2)))
        return frame_faces
    def store_image(self, frame_faces,path):
        for face in frame_faces:
            (x1, y1), (x2, y2) = frame_faces[face]
            # Region of Interest
            roi = self.frame[y1:y2, x1:x2]
            cv2.imwrite(path + "/" + face + ".jpg", roi)

    def get_video_frame_faces(self, video_string, saving_path):
        print("start:")
        print(datetime.now().strftime("%H:%M:%S"))
        vid_faces = {}
        vid = cv2.VideoCapture(video_string)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_number in range(0, total_frames, 4):
            self.count += 1
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = vid.read()
            if ret:
                self.frame = frame
                frame_faces = self.get_faces()
                if len(frame_faces) != 0:
                    self.store_image(frame_faces, saving_path)
                    vid_faces[self.count] = frame_faces

        # After the loop release the cap object
        vid.release()
        print("end:")
        print(datetime.now().strftime("%H:%M:%S"))