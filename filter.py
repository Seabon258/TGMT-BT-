import cv2
import numpy as np

# Đọc ảnh đầu vào
img = cv2.imread(r'filtersample.jpg')
image = img
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Cải thiện độ tương phản của ảnh bằng cách histogram equalization
gray = cv2.equalizeHist(gray)

# Resize ảnh xuống 50%
resized_img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

# Làm mượt ảnh với Gaussian blur
blurred = cv2.GaussianBlur(resized_img, (5, 5), 0)

# Chuyển ảnh sang không gian màu HSV
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# Xác định phạm vi màu xanh (nền) và trắng (mũi tên)
lower_blue = np.array([100, 150, 50])  # Màu xanh
upper_blue = np.array([140, 255, 255])

lower_white = np.array([0, 0, 200])  # Màu trắng
upper_white = np.array([255, 55, 255])

# Tạo mặt nạ cho màu xanh và màu trắng
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# Kết hợp mặt nạ xanh và trắng để chỉ giữ lại mũi tên trắng trên nền xanh
mask = cv2.bitwise_and(mask_white, mask_blue)

# Sử dụng Canny edge detection để làm nổi bật cạnh
edges = cv2.Canny(mask, 100, 200)

# Tìm contours trong ảnh đã xử lý với Canny
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Duyệt qua các contours và tính diện tích
for cnt in contours:
    area = cv2.contourArea(cnt)
    
    # Điều chỉnh diện tích sau khi resize (nếu ảnh được giảm 50%)
    adjusted_area = area * (1 / (0.5 * 0.5))  # Tính lại diện tích theo tỷ lệ resize

    print(f"Area after resizing: {adjusted_area}")

    # Thêm điều kiện lọc nếu cần
    if adjusted_area < 10:
        continue  # Bỏ qua những contour có diện tích nhỏ (nhiễu)
    
    # Vẽ contour và bounding box nếu diện tích hợp lệ
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.drawContours(resized_img, [cnt], -1, (0, 255, 0), 2)

# Hiển thị kết quả
cv2.imshow("Resized Image with Contours", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
