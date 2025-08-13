import os
import random
import shutil

# ==== CẤU HÌNH ====
image_dir = "image"   # Thư mục chứa toàn bộ ảnh gốc
label_dir = "label"   # Thư mục chứa toàn bộ label gốc
output_dir = "dataset"     # Thư mục output chuẩn YOLO
val_ratio = 0.2            # Tỉ lệ ảnh cho validation (20%)

# ==== TẠO CẤU TRÚC FOLDER ====
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# ==== LẤY DANH SÁCH ẢNH ====
images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
random.shuffle(images)

# ==== CHIA DATA ====
val_count = int(len(images) * val_ratio)
val_images = images[:val_count]
train_images = images[val_count:]

def copy_files(image_list, split):
    for img_name in image_list:
        # Copy ảnh
        shutil.copy(os.path.join(image_dir, img_name),
                    os.path.join(output_dir, "images", split, img_name))
        
        # Copy label (nếu có)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        if os.path.exists(label_path):
            shutil.copy(label_path,
                        os.path.join(output_dir, "labels", split, label_name))
        else:
            # Nếu không có label thì bỏ qua
            pass

# Copy train và val
copy_files(train_images, "train")
copy_files(val_images, "val")

print(f"✅ Done! Train: {len(train_images)}, Val: {len(val_images)}")
