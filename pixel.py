import cv2

# Đọc ảnh đầu vào
img = cv2.imread(r'filtersample.jpg')

# Kiểm tra kích thước ảnh (chiều cao và chiều rộng)
height, width, _ = img.shape

print(f"Kích thước ảnh: {width} x {height} pixels")
