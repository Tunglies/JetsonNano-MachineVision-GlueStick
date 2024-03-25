yolo export ^
    model=./data/weights/best.pt ^
    format=onnx ^
    imgsz=720,720 ^
    opset=15 ^
    half=True ^
    simplify=True