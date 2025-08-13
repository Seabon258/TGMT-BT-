# TGMT-BT-
file này làm bài tập sau khi đã lấy data và train
Quy trình thực hiện 

- Chuẩn bị data : quay video sau đó dung python cắt thành frame ra để lấy ảnh làm data gốc file get data
- Gán nhãn : make sense ai - 1 tool trên web để gán nhãn vào vật thể
( code trong get data có vấn đề khi import vào make sense vị trí sẽ bị đảo lộn vì vậy cần khác phục và đổi tên lại cho chuẩn : arrange.y)
sau khi gán nhãn xong thì xuất file yolo ra 
- Tạo file .yaml( chính là file data để huấn luyện ) trong đó có 2 cách làm cách thủ công tạo ra từng folder và với cấu trúc như sau :
   ├── images/
 │    ├── train/
 │    ├── val/
 ├── labels/
 │    ├── train/
 │    ├── val/
sau đó chọn thủ công ảnh để bỏ vào
train là ảnh huấn luyện
val là ảnh để test lại
hoặc cách 2 là dùng code để chia sep.py
- Dùng checkgpu.py để xem có gpu hỗ trợ không ( gpu nhanh hơn cpu trong việc này rất nhiều )
Bắt đầu train train.py và ngồi đợi :))
- check.py để xem kết quả đúng hay không ( có demo trong thư mục )
Đi đá cốc bia lạnh rồi ngủ.
