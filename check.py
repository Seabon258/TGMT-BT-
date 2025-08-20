from ultralytics import YOLO
from pathlib import Path
import cv2
import torch

device = 0 if torch.cuda.is_available() else "cpu"

# Load model
train_dir = Path("runs/train")
latest_exp = max(train_dir.iterdir(), key=lambda p: p.stat().st_mtime)
best_weights = latest_exp / "weights" / "best.pt"
model = YOLO(best_weights)
print("Using weights:", best_weights)

# Folder ảnh test
test_dir = Path("test")
test_images = list(test_dir.glob("*.jpg"))  # hoặc *.png

for img_path in test_images:
    results = model.predict(source=str(img_path), conf=0.5, save=True, device=device)
    result_img = results[0].plot()  # plot lên ảnh

    # Show ảnh
cv2.imshow("Prediction", result_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
