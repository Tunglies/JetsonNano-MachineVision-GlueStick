from src.logger import logging
import cv2
import nanocamera


class Camera(nanocamera.Camera):
    def __init__(self, camera_type=0, device_id=0, source="localhost:8080", flip=0, width=640, height=480, fps=30, enforce_fps=False, debug=False):
        super().__init__(camera_type, device_id, source, flip, width, height, fps, enforce_fps, debug)
        
    def __enter__(self):
        logging.info("Opening CSI Camera")
        return self
    
    def __exit__(self, arg0, arg1, arg2):
        logging.info("Closing CSI Camera")
        self.release()
        cv2.destroyAllWindows()
    
    def __wait_keys(self):
        key_code = cv2.waitKey(10) & 0xFF
        if key_code == 27 or key_code == ord("q"):
            self.release()
            cv2.destroyAllWindows()
    
    def stream(self):
        while True:
            yield self.read()
    
    def imshow(self, frame):
        self.__wait_keys()
        cv2.imshow("CSI Camera", frame)