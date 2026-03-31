import torch
# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version (used by PyTorch):", torch.version.cuda)
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version(PyTorch):", torch.version.cuda)
    print("GPU number:", torch.cuda.device_count())
    print("current GPU name:", torch.cuda.get_device_name(0))

import tiktoken
