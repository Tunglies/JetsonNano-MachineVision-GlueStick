from src.logger import logging
import cv2


class Camera:
    def __init__(
        self,
        flip_method: int = 0,
        windows_title: str = None,
        USB: int = False
    ) -> None:
        self.__windows_title: str = windows_title or "CSI Camera" if not USB else "USB Camera"
        logging.info("Setting Camera Device: %s", "CSI" if not USB else "USB")
        self.__video = (
            cv2.VideoCapture(
                self.__gstreamer_pipeline(flip_method=flip_method), cv2.CAP_GSTREAMER
            )
            if not USB
            # else cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else cv2.VideoCapture("/dev/video1")
        )
        logging.debug("Checking Camera Open: %s", self.__video.isOpened())
        if not self.__video.isOpened():
            logging.warning("Camera is not opened, exiting...")
            exit(1)

        logging.info("Setting Windows Name: %s", self.__windows_title)
        self.__set_window()

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        self.__video.release()
        cv2.destroyAllWindows()

    def __gstreamer_pipeline(
        self,
        sensor_id=0,
        capture_width=1920,
        capture_height=1080,
        display_width=960,
        display_height=540,
        framerate=30,
        flip_method=0,
    ):
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )

    def __set_window(
        self,
    ):
        cv2.namedWindow(winname=self.__windows_title, flags=cv2.WINDOW_AUTOSIZE)

    def __waitkeys(self):
        key_code = cv2.waitKey(10) & 0xFF
        if key_code == 27 or key_code == ord("q"):
            self.__video.release()
            cv2.destroyAllWindows()

    def stream(self):
        while True:
            ret, frame = self.__video.read()
            yield ret, frame

    def imshow(self, frame):
        self.__waitkeys()
        cv2.imshow(self.__windows_title, frame)