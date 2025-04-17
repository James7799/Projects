import cv2
import time
import torch
import argparse
import os
from ultralytics import YOLO

class Custom_detector:
    def __init__(self, device):
        self.device = device

    def detector(self, source, save_path="output.mp4"):
        device = self.device
        model = YOLO("yolo11n.pt").to(device)
        model.eval()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Video is not opened: {source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        out_name = f"{source.split('.')[0]}_proccessed_{save_path}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_name, fourcc, fps, (width, height))

        print(f"Writing started: {out_name}")

        prev_time = 0
        frame_count = 0

        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame or reached end of video.")
                break

            results = model.track(
                source=frame,
                classes=[0],
                tracker="bytetrack.yaml",
                persist=True,
                device=device,
                conf=0.4,
                iou=0.5,
                stream=True
            )

            for r in results:
                im = r.orig_img.copy()
                boxes = r.boxes

                if boxes is not None and len(boxes) > 0:
                    for box in boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Thin rectangle

            # FPS and latency
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            fps_disp = 1 / (end_time - prev_time + 1e-5)
            prev_time = end_time

            text = f"FPS: {fps_disp:.2f}  Latency: {latency:.2f}ms"
            cv2.putText(im, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Detection Frame', im)


            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): 
                print("Stopping execution...")
                break 

         
            out.write(im)
            frame_count += 1

        print(f"{frame_count} frames written. File: {out_name}")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def run(self, source, save_path):
        self.detector(source=source, save_path=save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or 0 to use webcam")
    parser.add_argument("--save_path", default="output.mp4", help="Output video file path.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = Custom_detector(device)
    try:
        source = int(args.video_path) if args.video_path == "0" else args.video_path
    except ValueError:
        source = args.video_path

    detector.run(source, args.save_path)
