This project turns a regular drone into an AI-powered guide drone that can recognize and track people or objects in real time.
Using a Raspberry Pi, a camera module, and the YOLOv5 object-detection model, you can build a compact, intelligent drone that “sees” its surroundings.

Overview

The Guide Drone uses computer vision to identify and follow specific objects (like people) while flying.
The onboard Raspberry Pi runs YOLOv5 to process the camera feed, detect objects, and send navigation cues to the flight controller.

Features

Real-time object detection using YOLOv5

Works with Raspberry Pi Camera Module or USB webcam

Compatible with ArduPilot or Betaflight drones

Lightweight, portable, and easy to replicate

Optional “Follow Me” mode for autonomous tracking

Hardware Requirements
Component	Description
Raspberry Pi 4/5	Main processor for running YOLOv5
Camera	Pi Camera Module v2 or compatible USB camera
Drone Frame	Any frame that supports Raspberry Pi (e.g., iFlight AOS 4", GEPRC Mark 4 5")
Power Supply	LiPo Battery (5 V output for Pi via BEC)
Flight Controller	ArduPilot / Betaflight compatible FC
Wi-Fi or LTE dongle	For remote access or telemetry (optional)
Software Requirements

Raspberry Pi OS (Lite recommended)

Python 3.8+

PyTorch + YOLOv5

OpenCV 4+

Drone SDK (optional: MAVLink, DroneKit, or PySerial)


Setup Steps

Flash Raspberry Pi OS

sudo apt update && sudo apt upgrade -y


Clone YOLOv5

git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt


Connect Camera

libcamera-still -o test.jpg


(Make sure your camera works before running detection.)

Run YOLOv5 Detection

python detect.py --source 0 --weights yolov5s.pt --conf 0.4


This runs real-time detection on the live camera feed.

Integrate with Drone

Use MAVLink / DroneKit Python to send position or velocity commands

Example:

vehicle.simple_goto(location_object_detected)


(Optional) Add tracking logic
Use bounding-box center coordinates from YOLO to move the drone toward the object.

How It Works

Camera captures live video

YOLOv5 detects objects in each frame

The Pi calculates relative position of the target

Navigation commands are sent to the drone’s flight controller

Example Output

When a person is detected, the console displays:

person 0.85 (x:320, y:210, w:120, h:230)


You can map this position to control yaw/pitch adjustments so the drone keeps the person centered.

Tips

Use the yolov5n (nano) model for faster performance on the Raspberry Pi

Run at 640×480 resolution for smoother FPS

If latency is high, use a remote inference server (e.g., Jetson Nano or PC)
