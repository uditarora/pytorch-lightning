import torch
import torch.nn as nn

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.core.memory import ModelSummary
from benchmarks.test_rnn_parity import ParityRNN


# TODO:
# Empty LightningModule (no layers)
# Device (CPU, GPU, amp)
# Different input shapes (tensor, nested lists, nested tuples, unknowns)

def test_linear_model_summary_shapes():
    """ Test that the model summary correctly computes the input- and output shapes. """

    class CurrentModel(LightningModule):

        def __init__(self):
            super().__init__()
            # note: the definition order is intentionally scrambled for this test
            self.layer2 = nn.Linear(10, 2)
            self.combine = nn.Linear(7, 9)
            self.layer1 = nn.Linear(3, 5)
            self.relu = nn.ReLU()
            self.unused = nn.Conv2d(1, 1, 1)

            self.example_input_array = (torch.rand(2, 3), torch.rand(2, 10))

        def forward(self, x, y):
            out1 = self.layer1(x)
            out2 = self.layer2(y)
            out = self.relu(torch.cat((out1, out2), 1))
            out = self.combine(out)
            return out

    model = CurrentModel()
    summary = ModelSummary(model)
    assert summary.in_sizes == [
        [2, 10],    # layer 2
        [2, 7],     # combine
        [2, 3],     # layer 1
        [2, 7],     # relu
        'unknown'
    ]
    assert summary.out_sizes == [
        [2, 2],     # layer 2
        [2, 9],     # combine
        [2, 5],     # layer 1
        [2, 7],     # relu
        'unknown'
    ]


def test_rnn_summary_shapes():
    model = ParityRNN()

    b = 3
    t = 5
    i = model.rnn.input_size
    h = model.rnn.hidden_size
    o = model.linear_out.out_features

    model.example_input_array = torch.zeros(b, t, 10)

    summary = ModelSummary(model)
    assert summary.in_sizes == [
        [b, t, i],  # rnn
        [b, t, h],  # linear
    ]
    assert summary.out_sizes == [
        [[b, t, h], [[1, b, h], [1, b, h]]],    # rnn
        [b, t, o]                               # linear
    ]


if __name__ == '__main__':
    test_rnn_summary_shapes()