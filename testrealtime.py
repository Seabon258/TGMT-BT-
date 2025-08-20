import cv2
from pathlib import Path
import torch
from ultralytics import YOLO

# Chọn device
device = 0 if torch.cuda.is_available() else "cpu"

# Load model từ weights tốt nhất
train_dir = Path("runs/train")
latest_exp = max(train_dir.iterdir(), key=lambda p: p.stat().st_mtime)
best_weights = latest_exp / "weights" / "best.pt"
model = YOLO(best_weights)
print("Using weights:", best_weights)

# Open video
video_path = r"G:\hk41\xla\video.mp4"  # raw string để tránh lỗi \
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Resize frame nếu quá lớn để tăng tốc
    frame_resized = cv2.resize(frame, (640, 640))

    # YOLO inference
    results = model(frame_resized, device=device)

    # Vẽ kết quả lên frame
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("YOLO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
