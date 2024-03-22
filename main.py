import sys
from src import camera
from src import interface


USB = False
command = sys.argv
if len(command) > 1:
    if command[1] == "DEBUG":
        USB = True 

with camera.Camera(USB=USB) as video:
    model = interface.Onnx()
    for ret, frame in video.stream():
        pred = model.pre_process(frame)
        predict = model.predict(pred)
        output = model.post_process(predict, frame)
        if output:
            video.imshow(output)
            continue
        video.imshow(frame)
            