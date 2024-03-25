import cv2
import nanocamera as nano
# Create the Camera instance for No rotation (flip=0) with size of 640 by 480
camera = nano.Camera(camera_type=1, device_id=1, width=640, height=480, fps=30)

while camera.isReady():
    try:
        frame = camera.read()
        cv2.imshow("Video Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    except KeyboardInterrupt:
            break
    
    camera.release()

    del camera