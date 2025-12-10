# ® Betounis Robotics
# 2025
# kakoshund@yahoo.com

import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, filedialog
import time

Tk().withdraw()

# Video selection
left_video_path = filedialog.askopenfilename(title="Select Left Camera Video")
right_video_path = filedialog.askopenfilename(title="Select Right Camera Video")

# Baseline & focal length input
baseline_cm = float(input("Distance between cameras (baseline) in cm: "))
focal_length_px = float(input("Focal length (in pixels): "))

# Load YOLO model
model = YOLO("model_fish_tdd_v2_yl11s.pt")

cap_left = cv2.VideoCapture(left_video_path)
cap_right = cv2.VideoCapture(right_video_path)

def get_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

# Stored disparities from first 30 seconds
stored_disparities = []

start_time = time.time()
stereo_windows_closed = False

while True:
    retL, frameL = cap_left.read()

    # Stop when left video ends
    if not retL:
        break

    elapsed = time.time() - start_time

    # --------------------------------------------------------------------
    # 1. First 30 seconds → stereo mode (2 cameras, 2 windows)
    # --------------------------------------------------------------------
    if elapsed <= 30: # declare of 30 sec
        retR, frameR = cap_right.read()
        if not retR:
            break

        # YOLO detections
        results_left = model(frameL, verbose=False)[0]
        results_right = model(frameR, verbose=False)[0]

        fish_left = []
        fish_right = []

        # Left camera
        for det in results_left.boxes:
            x1, y1, x2, y2 = det.xyxy[0]
            cx, cy = get_center((x1, y1, x2, y2))
            fish_left.append((cx, cy, (int(x1), int(y1), int(x2), int(y2))))

        # Right camera
        for det in results_right.boxes:
            x1, y1, x2, y2 = det.xyxy[0]
            cx, cy = get_center((x1, y1, x2, y2))
            fish_right.append((cx, cy, (int(x1), int(y1), int(x2), int(y2))))

        # --- Fish matching ---
        for (cxL, cyL, boxL) in fish_left:
            best_match = None
            min_dist = 999999
            for (cxR, cyR, boxR) in fish_right:
                d = abs(cxL - cxR) + abs(cyL - cyR)
                if d < min_dist:
                    min_dist = d
                    best_match = (cxR, cyR, boxR)

            if best_match is None:
                continue

            cxR, cyR, boxR = best_match
            disparity = abs(cxL - cxR)

            if disparity >= 1:
                stored_disparities.append(disparity)

            # Draw box on left frame
            x1, y1, x2, y2 = boxL
            cv2.rectangle(frameL, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Left (Stereo Mode 0-30s)", frameL)
        cv2.imshow("Right (Stereo Mode 0-30s)", frameR)

    # --------------------------------------------------------------------
    # 2. After 30 seconds → ONLY left camera, 1 window
    # --------------------------------------------------------------------
    else:
        # Close both old windows ONCE
        if not stereo_windows_closed:
            cv2.destroyAllWindows()   # Close both stereo windows
            cap_right.release()       # Stop right camera
            stereo_windows_closed = True
            print("➡ Stereo mode windows closed. Continuing with monocular detection.")

        # YOLO only on left frame
        results_left = model(frameL, verbose=False)[0]

        if len(stored_disparities) == 0:
            avg_disparity = None
        else:
            avg_disparity = sum(stored_disparities) / len(stored_disparities)

        for det in results_left.boxes:
            x1, y1, x2, y2 = det.xyxy[0]
            cxL, cyL = get_center((x1, y1, x2, y2))

            # Depth estimation using mean disparity
            if avg_disparity is None or avg_disparity < 1:
                text = "No depth learned"
            else:
                distance = (baseline_cm * focal_length_px) / avg_disparity
                text = f"{distance:.2f} cm"

            cv2.rectangle(frameL,
                          (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 0, 0), 2)
            cv2.putText(frameL, text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)

        cv2.imshow("Left (Depth Mode)", frameL)

    # Exit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
