import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Variable whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Variable.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, input):
        inputs = [feat.squeeze(2) for feat in input.split(1, dim=2)]
        assert len(self) == len(inputs)
        outputs = [f(x) for f, x in zip(self, inputs)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs

class PositionwiseFeedForward(nn.Module):
	""" A two-layer Feed-Forward-Network with residual layer norm.

		Args:
			size (int): the size of input for the first-layer of the FFN.
			hidden_size (int): the hidden layer size of the second-layer
							  of the FNN.
			dropout (float): dropout probability(0-1.0).
	"""
	def __init__(self, size, hidden_size, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(size, hidden_size)
		self.w_2 = nn.Linear(hidden_size, size)
		self.layer_norm = LayerNorm(size)
		# Save a little memory, by doing inplace.
		self.dropout_1 = nn.Dropout(dropout)
		self.relu = nn.ReLU()
		self.dropout_2 = nn.Dropout(dropout)

	def forward(self, x):
		inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
		output = self.dropout_2(self.w_2(inter))
		return output + x

class FeedForward(nn.Module):
	""" A two-layer Feed-Forward-Network with residual layer norm.

		Args:
			size (int): the size of input for the first-layer of the FFN.
			hidden_size (int): the hidden layer size of the second-layer
							  of the FNN.
			dropout (float): dropout probability(0-1.0).
	"""
	def __init__(self, size, dropout=0.1):
		super(FeedForward, self).__init__()
		self.w_1 = nn.Linear(size, size)
		#self.w_2 = nn.Linear(hidden_size, size)
		self.layer_norm = LayerNorm(size)
		# Save a little memory, by doing inplace.
		self.dropout_1 = nn.Dropout(dropout, inplace=True)
		self.relu = nn.ReLU(inplace=True)
		#self.dropout_2 = nn.Dropout(dropout)

	def forward(self, x):
		inter = self.relu(self.w_1(self.layer_norm(x)))
		output = self.dropout_1(inter)
		return output + x
