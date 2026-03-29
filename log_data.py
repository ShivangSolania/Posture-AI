import csv
import time

def log(name, neck_angle, spine_angle, avg_neck, avg_spine, dev_detected, risk):
    with open(name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            neck_angle,
            spine_angle,
            avg_neck,
            avg_spine,
            dev_detected,
            risk
        ])