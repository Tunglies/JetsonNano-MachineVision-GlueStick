from src import camera
from src import interface


model = interface.Onnx(conf_threshold=0.35, resize_shape=(720, 720))
with camera.Camera(USB=False) as video:
    for ret, frame in video.stream():
        pred = model.pre_process(frame)
        predict = model.predict(pred)
        output = model.post_process(predict, frame)
        video.imshow(output)            