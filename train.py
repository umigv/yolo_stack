from ultralytics import YOLO
from roboflow import Roboflow
import os
import torch
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    print("mps not avalible")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

rf = Roboflow(api_key="lsvvGPgOaqWqgMgoZWKS")
project = rf.workspace("arv-ysash").project("02.03.2024-lane-lines-only")
version = project.version(6)
dataset = version.download("yolov8")

path = os.getcwd()


# Load a model
model = YOLO('LLOnly180ep.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data=path+"/yolo_project/data.yaml", epochs=100, imgsz=640, device=torch.device("mps"))