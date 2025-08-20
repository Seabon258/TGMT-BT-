from ultralytics import YOLO
import torch
from pathlib import Path

if __name__ == '__main__':
    # --- Kiểm tra GPU ---
    if torch.cuda.is_available():
        print(" GPU detected:", torch.cuda.get_device_name(0))
        device = 0  # GPU ID 0
    else:
        print(" GPU not detected. Using CPU.")
        device = "cpu"

    # --- Load model pretrained YOLOv8n ---
    model = YOLO("yolov8n.pt")  

    # --- Train model ---
    project_dir = Path("./runs/train")  
    model.train(
        data="data.yaml",      
        epochs=500,           
        imgsz=640,            
        batch=16,              
        device=device,        
        project=project_dir,  
        name="exp1"            
    )

    # --- Lấy đường dẫn best.pt ---
    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\n Training finished! Best weights saved at: {best_weights.resolve()}")
