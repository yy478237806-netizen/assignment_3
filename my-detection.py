from ast import Store
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
import jetson_inference
import jetson_utils

net = detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = videoSource("~/Pictures/horse_1.jpeg","~/Pictures/horse_0.jpeg") 
display = videoOutput("display://") 

while display.IsStreaming():
  img = camera.Capture()

  if img is None: 
    continue

  detections = net.Detect(img)
  print(detections)

  display.Render(img)
  display.SetStatus("Object Detection | Network {:.100f} FPS".format(net.GetNetworkFPS()))
  
