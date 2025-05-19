import keras

from src.models.video_capturer import VideoCapturer

mask_model = keras.models.load_model("../src/models/mask_detection_model.keras")
videoCapturer = VideoCapturer(0,60,mask_model, height=600, width=600)

videoCapturer.start_capturing()