from src import camera
from src import interface


# model = interface.Onnx(conf_threshold=0.25, resize_shape=(736, 736))

resize = (1080, 1080)

# model = interface.TensorRT(conf_threshold=0.45)
# def start(self):
#         self.cam_thread = Thread(target=self.__thread_read)
#         self.cam_thread.daemon = True
#         self.cam_thread.start()
#         return self

# model = interface.TensorRT()
with camera.Camera(flip=0, width=1920, height=1080, fps=30) as video:
    for frame in video.stream():
        # frame = model.detect(frame)
        video.imshow(frame)