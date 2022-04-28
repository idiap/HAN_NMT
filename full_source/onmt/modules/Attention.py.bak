"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


		H_1 H_2 H_3 ... H_n
		  q   q   q	   q
			|  |   |	   |
			  \ |   |	  /
					  .....
				  \   |  /
						  a

Constructs a unit mapping.
	$$(H_1 + H_n, q) => (a)$$
	Where H is of `batch x n x dim` and q is of `batch x dim`.

	The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import sys

class Attention(nn.Module):
	def __init__(self, dim):
		super(Attention, self).__init__()
		self.linear_in = nn.Linear(dim, dim, bias=False)
		self.sm = nn.Softmax(-1)
		self.linear_ctx = nn.Linear(dim, dim, bias=False)

	def forward(self, key, value, query, mask=None):
		"""
		query: batch x queryL x dim
		value: batch x sourceL x dim
		"""
		query = self.linear_in(query)  # batch x queryL x dim
		key = self.linear_ctx(key) # batch x sourceL x dim

		# Get attention
		attn = torch.bmm(query, key.transpose(1,2))  # batch x queryL x sourceL
		if mask is not None:
			attn = attn.masked_fill(Variable(mask), -1e18)
		attn = self.sm(attn)

		weightedContext = torch.bmm(attn, value)  # batch x dim
		
		return weightedContext, attn

class SimpleAttention(nn.Module):
	def __init__(self, dim):
		super(Attention, self).__init__()
		self.linear_in = nn.Linear(dim, dim, bias=False)
		self.sm = nn.Softmax(-1)
		self.linear_ctx = nn.Linear(dim, dim, bias=False)
		self.relu == nn.ReLU()

	def forward(self, key, value, mask=None):
		"""
		query: batch x queryL x dim
		value: batch x sourceL x dim
		"""
		attn = self.relu(self.linear_ctx(key)) # batch x sourceL x dim

		# Get attention
		#attn = torch.bmm(query, key.transpose(1,2))  # batch x queryL x sourceL
		if mask is not None:
			attn = attn.masked_fill(Variable(mask), -1e18)
		attn = self.sm(attn)

		weightedContext = torch.bmm(attn, value)  # batch x dim
		
		return weightedContext, attn
