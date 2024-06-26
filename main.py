import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 创建一个在CPU上的张量
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)

    # 将张量移动到GPU上
    x = x.cuda()

    # 在GPU上执行操作：将张量的每个元素乘以2
    y = x * 2

    # 将结果移回CPU
    y_cpu = y.cpu()

    # 打印结果
    print("Result on CPU:", y_cpu)
else:
    print("CUDA is not available. Please check your PyTorch installation and CUDA setup.")
