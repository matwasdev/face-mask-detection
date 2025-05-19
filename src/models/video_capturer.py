import cv2
import numpy as np

from src.exceptions.capture_error import CaptureError
from src.models.face_detector import FaceDetector


class VideoCapturer:
    def __init__(self, device, fps, mask_detection_model, width=300, height=300):
        self.width = width
        self.height = height
        self.mask_detection_model = mask_detection_model
        self.device = device
        self.fps = fps

    def initialize_cap(self):
        cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def start_capturing(self):
        face_detector = FaceDetector()
        cap = self.initialize_cap()

        while True:
            frame_status, frame = cap.read()

            if not frame_status:
                raise CaptureError("Something went wrong while capturing a frame")


            frame_colors_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_detected = face_detector.detect_faces(frame_colors_rgb)

            for face in faces_detected:
                x, y, width, height = face['box']

                face = frame[y:y + height, x:x + width]
                face_resized = cv2.resize(face, (32, 32))
                face_resized = np.array(face_resized).astype('float32') / 255.0
                face_resized = np.expand_dims(face_resized, axis=0)
                prediction = self.mask_detection_model.predict(face_resized, verbose=0)
                print("PREDICTION: " + str(prediction))

                label = "with_mask" if prediction > 0.5 else "without_mask"
                status_color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)

                confidence = round(prediction[0][0],3) if label == 'with_mask' else round(1 - prediction[0][0],3)



                cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.putText(frame, str(confidence), (x - 40, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
                cv2.rectangle(frame, (x, y), (x + width, y + height), status_color, 2)



            face_detector.faces_detected = faces_detected.__len__() if faces_detected.__len__() > 0 else 0
            cv2.putText(frame,"Detected faces: " + str(face_detector.faces_detected), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2 )

            cv2.imshow("Face Mask Detection", frame)
            if 0xFF & cv2.waitKey(1) == ord('q'):
                break