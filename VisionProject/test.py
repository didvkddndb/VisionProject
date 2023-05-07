import torch
print(torch.__version__)
a = torch.tensor([[1., -1.], [1., -1.]])
print(a)
print(torch.cuda.is_available())
if torch.cuda.is_available() == True:
    print(a.cuda())