import torch

class GEMM(torch.nn.Module):
    def __init__(self, K, N):
        super(GEMM, self).__init__()
        self.linear = torch.nn.Linear(K, N)

    def forward(self, x):
        return self.linear(x)

M, N, K = 4, 8, 16

x = torch.randn(M, K)
model = GEMM(K, N)

model.eval()
torch.onnx.export(model, x, 'test.onnx', input_names = ['input'], output_names = ['output'])

# pytorch: https://pytorch.org/tutorials/beginner/saving_loading_models.html
# torch.save(model.state_dict(), PATH)
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
