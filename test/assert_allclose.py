import numpy as np
import torch
import sys

sys.path.append(r"C:\Users\xxngu\Documents\sourcecode\minorgrad")
from minorgrad.tensor import Tensor

x_init = np.random.randn(1, 3).astype(np.float32)
W_init = np.random.randn(3, 3).astype(np.float32)
m_init = np.random.randn(1, 3).astype(np.float32)


def test_minorgrad():
    x = Tensor(x_init)
    W = Tensor(W_init)
    m = Tensor(m_init)

    out = x.dot(W)
    outr = out.relu()
    outl = outr.logsoftmax()
    outm = outl.mul(m)
    outa = outm.add(m)
    outx = outa.sum()

    outx.backward()
    return outx.data, x.grad, W.grad


def test_pytorch():
    x = torch.tensor(x_init, requires_grad=True)
    W = torch.tensor(W_init, requires_grad=True)
    m = torch.tensor(m_init)

    out = x.matmul(W)
    outr = out.relu()
    outl = torch.nn.functional.log_softmax(outr, dim=1)
    outm = outl.mul(m)
    outa = outm.add(m)
    outx = outa.sum()

    outx.backward()
    return outx.detach().numpy(), x.grad, W.grad


for minorgrad_value, pytorch_value, value_name in zip(
    test_minorgrad(), test_pytorch(), ["outx", "x.grad", "W.grad"]
):
    print(f"{value_name} minorgrad: {minorgrad_value}")
    print(f"{value_name} pytorch: {pytorch_value}")
    print()
    np.testing.assert_allclose(minorgrad_value, pytorch_value, atol=1e-6)
