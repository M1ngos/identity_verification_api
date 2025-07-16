# import torch
#
# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# x = torch.rand(3, 3)
# print("Random tensor:\n", x)


import torch
# print(torch.version.cuda)  # Will print CUDA version if GPU build, or None if CPU only
# print(torch.cuda.is_available())  # True if CUDA device detected


print(torch.__version__)
print(torch.cuda.is_available())