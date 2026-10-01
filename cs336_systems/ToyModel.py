import torch.nn as nn
import torch

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f" x dtype : {x.dtype}" )
        ff1 = self.fc1(x)
        print(f" fc1 dtype : {ff1.dtype}" )
        x = self.relu(ff1)
        print(f" relu dtype : {x.dtype}" )
        x = self.ln(x)
        print(f" ln dtype : {x.dtype}" )
        x = self.fc2(x)
        print(f" fc2 dtype : {x.dtype}" )
        return x
    

in_features = 100
batch = 4
device = 'mps'
model = ToyModel(in_features, 200).to(device)
x = torch.randn(batch,in_features, device=device)
targets = torch.randint(0, 200, (4,), device=device)  # class indices, (batch,)

with torch.autocast(device_type=device, dtype=torch.float16):
    print("params: ", {n: p.dtype for n, p in model.named_parameters()})
    y = model(x)
    print("logits: ", y.dtype)
    loss = nn.functional.cross_entropy(y, targets)
    print("loss:   ", loss.dtype)

loss.backward()                                     # backward goes outside autocast
print("grads:  ", {n: p.grad.dtype for n, p in model.named_parameters()})