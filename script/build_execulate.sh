# rm -rf ./build

    # --include-data-dir=./data/weights=./data/weights \
python3 -m nuitka \
    --clang \
    --standalone \
    --output-dir=./build \
    --jobs=4 \
    --follow-imports \
    --follow-stdlib \
    --follow-import-to=numpy \
    --nofollow-import-to=setuptools,doctest,cv2, \
    main.py

cp ./build/main.dist/main.bin .