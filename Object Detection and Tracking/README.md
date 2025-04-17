
# YOLOv11 + ByteTrack Human Tracking System

## Environment Setup

- **OS**: Windows 11  
- **GPU**: NVIDIA GTX 1650  
- **CPU**: Intel i5-11400H  
- **Framework**: Ultralytics YOLOv11  
- **Libraries**: OpenCV, PyTorch

## How to Run

1. Install dependencies:
```bash
pip install ultralytics opencv-python torch
```
2. Replace "bytetrack.yaml" file n your corresponding "ultralytics\cfg\trackers" folder with which provided in zip file.
3. Run the script with a video file or webcam:
```bash
python project_final.py --video_path video.mp4
```

Use `--video_path 0` for webcam input.

## Performance

- **FPS**: 16–17 average
- **Latency**: 21–22ms !!! These results were achieved on the specified hardware configuration

## Notes

- Uses `yolo11n.pt` (nano model) for real-time detection
- ByteTrack is configured via `bytetrack.yaml` for consistent ID assignment
- Suitable for future mobile deployment
