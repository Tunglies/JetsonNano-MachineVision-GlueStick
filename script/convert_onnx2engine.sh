/usr/src/tensorrt/bin/trtexec \
    --onnx=./data/weights/best.onnx \
    --saveEngine=./data/weights/best.engine \
    --best \
    --workspace=2048