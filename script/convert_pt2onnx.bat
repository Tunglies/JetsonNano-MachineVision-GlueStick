yolo export ^
    model=./data/weights/best.pt ^
    format=onnx ^
    imgsz=640,640 ^
    opset=16
    @REM half=true
