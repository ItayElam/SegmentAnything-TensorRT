FROM nvcr.io/nvidia/tensorrt:23.04-py3

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install matplotlib segment-anything torch torchvision opencv-python onnxruntime-gpu onnx tqdm prettytable
COPY $pwd/sam_modification/predictor.py /usr/local/lib/python3.8/dist-packages/segment_anything/predictor.py

