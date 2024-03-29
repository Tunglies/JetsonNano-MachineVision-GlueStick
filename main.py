import sys
sys.path.append("/usr/lib/python3.6/dist-packages")
sys.path.append("/usr/lib/python3/dist-packages")

import time
from src import utils
from src import camera
from src import interface
from queue import Queue
from threading import Thread, Timer


frame_queue = Queue()
result_queue = Queue()

model = interface.Onnx(
    onnx_weights_file="best_m.onnx",
    conf_threshold=0.2,
    resize_shape=(640, 640)
)
video = camera.Camera(flip=0, width=1920, height=1080, fps=30, enforce_fps=True)
status = interface.Status(pin=12)


def model_thread():
    while True:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        if frame is None:
            break

        frame, flag = model.detect(frame)
        status.set_status(flag)
        result_queue.put(frame)


def beep_thread():
    timer = None
    while True:
        if status.status:
            if timer is None:
                timer = Timer(3, status.beep)
                timer.start()
        else:
            if timer is not None:
                timer.cancel()
                timer = None
        time.sleep(0.1)


model_thread_instance = Thread(target=model_thread)
model_thread_instance.daemon = True
model_thread_instance.start()

beep_thread_instance = Thread(target=beep_thread)
beep_thread_instance.daemon = True
beep_thread_instance.start()

for frame in video.stream():
    frame_queue.put(frame)
    
    if result_queue.empty():
        video.imshow(frame)

    result = result_queue.get()
    if result is None:
        break

    video.imshow(result)


utils.Utils.clean_garbages(
    status,
    beep_thread_instance,
    video,
    model_thread_instance
)