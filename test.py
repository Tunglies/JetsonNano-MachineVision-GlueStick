from src import camera
with camera.Camera() as video:
    for frame in video.stream():
        video.imshow(frame)