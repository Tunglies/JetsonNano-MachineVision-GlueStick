import cv2

# 定义视频捕获对象
cap = cv2.VideoCapture('/dev/video1')  # 使用你的摄像头设备地址

# 检查摄像头是否成功开启
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 逐帧捕获
    ret, frame = cap.read()

    # 如果正确读取帧，ret为True
    if not ret:
        print("无法读取帧")
        break

    # 显示结果帧
    cv2.imshow('Frame', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放捕获器和关闭所有窗口
cap.release()
cv2.destroyAllWindows()
