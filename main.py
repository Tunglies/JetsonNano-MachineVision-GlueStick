from src import camera
from src import interface
from queue import Queue
from threading import Thread


frame_queue = Queue()
result_queue = Queue()


model = interface.Onnx(
    onnx_weights_file="best_m.onnx",
    conf_threshold=0.2,
    resize_shape=(640, 640)
)
video = camera.Camera(flip=0, width=1920, height=1080, fps=30, enforce_fps=True)


def model_thread():
    while True:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        if frame is None:
            break

        frame = model.detect(frame)
        result_queue.put(frame)


m = Thread(target=model_thread)
m.daemon = True
m.start()


for frame in video.stream():
    frame_queue.put(frame)
    
    if result_queue.empty():
        video.imshow(frame)

    result = result_queue.get()
    if result is None:
        break

    video.imshow(result)