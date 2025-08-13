from ultralytics import YOLO
from pathlib import Path
import cv2
import torch

train_dir = Path("runs/train")
latest_exp = max(train_dir.iterdir(), key=lambda p: p.stat().st_mtime)
best_weights = latest_exp / "weights" / "best.pt"

print("Using weights:", best_weights)


device = 0 if torch.cuda.is_available() else "cpu"


model = YOLO(best_weights)


results = model.predict(source="frame1093.jpg", conf=0.5, save=True, device=device)

result_img = results[0].plot()


cv2.imshow("Prediction", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite("result.jpg", result_img)
