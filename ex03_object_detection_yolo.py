"""Ex 3: Object detection using YOLOv5.

Usage: python ex03_object_detection_yolo.py path/to/image.jpg
"""

import sys

import matplotlib.pyplot as plt
import torch

source = sys.argv[1] if len(sys.argv) > 1 else input("Image path or URL: ")

# Pretrained YOLOv5-small; confidence threshold 0.25, input size 640 (as in the lab)
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
model.conf = 0.25

results = model(source, size=640)
results.print()
results.save()  # saves annotated image under runs/detect/exp*

plt.imshow(results.render()[0])
plt.axis("off")
plt.show()
