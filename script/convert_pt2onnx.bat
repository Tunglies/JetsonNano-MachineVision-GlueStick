yolo export ^
    model=./data/weights/best.pt ^
    format=onnx ^
    imgsz=640,640 ^
    opset=15 ^
    half=True ^
    simplify=True