import torch
print("PyTorch版本：", torch.__version__)
print("CUDA是否可用：", torch.cuda.is_available())
print("测试张量创建：", torch.tensor([1,2,3]))