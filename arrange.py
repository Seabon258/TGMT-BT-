import os

folder = r"G:\hk41\xla\image"  # Đường dẫn tới thư mục ảnh
files = sorted(os.listdir(folder), key=lambda x: int(x.replace("frame", "").replace(".jpg", "")))

for i, filename in enumerate(files):
    new_name = f"frame{i:04d}.jpg"  # Định dạng 4 số
    os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))

print("✅ Đã rename xong!")
