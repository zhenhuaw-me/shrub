import torch

class Conv(torch.nn.Module):
    def __init__(self, IC, OC, KS):
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv2d(IC, OC, KS, bias=False)

    def forward(self, x):
        return self.conv(x)

N, IC, IH, IW = 1, 2, 8, 8
OC, KS = 4, (3, 3)

x = torch.randn(N, IC, IH, IW)
model = Conv(IC, OC, KS)

model.eval()
torch.onnx.export(model, x, 'test.onnx', input_names = ['input'], output_names = ['output'])

# pytorch: https://pytorch.org/tutorials/beginner/saving_loading_models.html
# torch.save(model.state_dict(), PATH)
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))

# import shrub
# shrub.onnx.run('test.onnx')
