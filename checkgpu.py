import torch
print(torch.cuda.is_available())  # True nếu GPU dùng được
print(torch.cuda.get_device_name(0))  # tên GPU
