import cv2
import numpy as np
from typing import Optional

class BackgroundSubtractor:
    def __init__(self, video_path: str, history: int = 100, var_threshold: int = 25, detect_shadows: bool = True):
        self.video_path = video_path
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.cap: Optional[cv2.VideoCapture] = None
        self.subtractor: Optional[cv2.BackgroundSubtractor] = None

    def _initialize_capture(self) -> None:
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=self.var_threshold, detectShadows=self.detect_shadows
        )

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.subtractor.apply(frame)

    def run(self, frame_delay: int = 20) -> None:
        self._initialize_capture()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            mask = self._process_frame(frame)
            cv2.imshow("Video Frame", frame)
            cv2.imshow("Foreground Mask", mask)
            if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    VIDEO_PATH = "video.mp4"
    bg_subtractor = BackgroundSubtractor(VIDEO_PATH)
    bg_subtractor.run()
