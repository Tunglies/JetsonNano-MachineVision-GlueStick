from src import camera
from src import interface



resize_shape = (640, 640)
model = interface.Onnx(conf_threshold=0.25, resize_shape=resize_shape)
video =  camera.Camera(flip=0, width=1920, height=1080, fps=30, enforce_fps=True)
for frame in video.stream():
    frame = model.detect(frame)
    video.imshow(frame)