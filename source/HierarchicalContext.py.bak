### Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
### Written by Lesly Miculicich <lmiculicich@idiap.ch>

# Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
# Written by Lesly Miculicich <lmiculicich@idiap.ch>
# 
# This file is part of HAN-NMT.
# 
# HAN-NMT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# HAN-NMT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with HAN-NMT. If not, see <http://www.gnu.org/licenses/>.

import torch, sys
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy

import onmt
from onmt.Models import EncoderBase
from onmt.Models import DecoderState
from onmt.Utils import aeq
from onmt.modules.Embeddings import PositionalEncoding
from onmt.modules.Attention import Attention
from onmt.modules import PositionwiseFeedForward
from onmt.modules import FeedForward

class HierarchicalContext(nn.Module):
	"""
	Hierarquical module for OpenNMT

	Args:
	  size: hidden layers size
	  dropout: dropout for each layer
	  head_count: number of heads for the multi-head attention
	  hidden_size: Position-wise Feed-Forward hidden layers size
	  context_size: number of previous sentences
	  padding_idx: id for padding word
	"""

	def __init__(self, size, dropout,
				 head_count=8, hidden_size=2048, 
				 context_size=3, padding_idx=1):
		super(HierarchicalContext, self).__init__()
		
		self.context_size = context_size
		self.padding_idx = padding_idx

		self.layer_norm_query_word = onmt.modules.LayerNorm(size)
		self.layer_norm_query_sent = onmt.modules.LayerNorm(size)
		self.layer_norm_word = onmt.modules.LayerNorm(size)
		self.layer_norm_sent = onmt.modules.LayerNorm(size)

 		self.dropout = nn.Dropout(dropout)

		self.sent_attn = onmt.modules.MultiHeadedAttention(head_count, size, dropout=dropout)
		self.word_attn = onmt.modules.MultiHeadedAttention(head_count, size, dropout=dropout)

		self.linear = nn.Linear(2*size, size)
		self.sigmoid = nn.Sigmoid()

		self.feed_forward = PositionwiseFeedForward(size,
												hidden_size,
												dropout)

		
	"""
	Function to reshape and select vectors for word-level attention

	Args:
	  query: 	(FloatTensor) encoder/decoder hidden states of current sentence 
					{batch_1 x sentence_lenght_1 x dim_hidden_state}
	  context: 	(FloatTensor) encoder/decoder hidden states of previous sentences 
					{batch_2 x sentence_lenght_2 x dim_hidden_state}
	  index: 	(LongTensor) matrix that indicates the index of the context for each example in the query 
					{batch_1 x context_size} 
				For example: Having batch_1=2 and context_size=3, index=[[0,1,2], [1,2,3]] . 
							 Here we take the sentence 0, 1 and 2 from context for the first query, and 
							 1, 2 and 3 from context for the second query. 
      mask_word:(ByteTensor) context word padding mask
					{batch_2 x sentence_lenght_2 x 1}

	Out:
	  query_:	(FloatTensor) query reshaped to facilitate future operations 
					{batch_1*context_size x sentence_lenght_1 x dim_hidden_state}
      context_word: (FloatTensor) context filtered and reshaped to facilitate future operations 
					{batch_1*context_size x sentence_lenght_2 x dim_hidden_state}
 	  context_pad_mask: (ByteTensor) padding mask for context_word 
					{batch_1*context_size x sentence_lenght_1 x sentence_lenght_2}
	"""
	def _get_word_context(self, query, context, index, mask_word):

		"""  Verify sizes  """
		b_size, t_size, d_size = query.size()
		b_size_, s_size, d_size_ = context.size()
		aeq(d_size, d_size_)
		b_size__, c_size = index.size()
		aeq(b_size, b_size__)
		b_size__, t_size_, s_size_ = mask_word.size()
		aeq(b_size_, b_size__)
		aeq(s_size, s_size_)
		aeq(t_size, t_size_)
	
		"""  Padding index of previous invalid sentences index (<0) to 0, and saving mask for sentences  """
		mask_sent = index < 0
		index_ = copy.deepcopy(index)
		index_[mask_sent] = 0

		"""  Select context with index vector  """ 
		context_ = context.view(b_size_, -1).expand(b_size, b_size_, s_size*d_size)
		index__ = index_.unsqueeze(2).expand(b_size , c_size, s_size*d_size)
		context_word = torch.gather(context_, 1, Variable(index__, 
					requires_grad=False)).view(b_size*c_size, s_size, d_size)


		"""  Create complete mask for context: word padding + sentence padding """
		mask_ = mask_word.contiguous().view(b_size_, -1).expand(b_size, b_size_, t_size_*s_size)
		index__ = index_.unsqueeze(2).expand(b_size, c_size, t_size_*s_size)
		context_pad_mask = torch.gather(mask_, 1, index__).view(b_size*c_size, t_size_, s_size)
		mask_sent_ = mask_sent.unsqueeze(2).expand(b_size, 
					c_size, t_size_*s_size).contiguous().view(b_size*c_size, t_size_, s_size)
		context_pad_mask[mask_sent_] = self.padding_idx
		
		"""  Reshape query for future operations  """  
		query_ = query.unsqueeze(1).expand(b_size , c_size, 
					t_size, d_size).contiguous().view(b_size*c_size, t_size, d_size)
		
		return query_, context_word, context_pad_mask 




	"""
	Function to reshape and select vectors for sentence-level attention

	Args:
	  query:	(FloatTensor) encoder/decoder hidden states of current sentence 
					{batch_1 x sentence_lenght_1 x dim_hidden_state}
	  context_word: (FloatTensor) encoder/decoder hidden states of the attention to words 
					{batch_1*context_size x sentence_lenght_1 x dim_hidden_state}
	  index:	(LongTensor) matrix that indicates the index of the context for each example in the query 
					{batch_1 x context_size}
				For example: Having batch_1=2 and context_size=3, index=[[0,1,2], [1,2,3]] . 
							 Here we take the sentence 0, 1 and 2 from context for the first query, and 
							 1, 2 and 3 from context for the second query. 
      attn_word:(FloatTensor) word-level attention weights 
					{batch_1*context_size x heads x sentence_lenght_1 x sentence_lenght_2}

	Out:
	  query_: (FloatTensor) reshaped query  
				{batch_1*sentence_lenght_1 x 1 x dim_hidden_state}
      context_sent: (FloatTensor) filtered and reshaped context 
				{batch_1*sentence_lenght_1 x context_size x dim_hidden_state}
 	  context_pad_mask: (ByteTensor) padding mask for context_sent 
				{batch_1*sentence_lenght_1 x 1 x context_siz}
	  attn_word: (FloatTensor) reshaped word-level attention weights  
				{batch_1 x context_size x heads x sentence_lenght_1 x sentence_lenght_2}
	"""
	def _get_sent_context(self, query, context_word, index, attn_word):

		b_size, t_size, d_size = query.size()
		_, c_size = index.size()	

		"""  Reshape context from word-level attention  """  
		context_sent = context_word.view(b_size, c_size, t_size, d_size).transpose(1,2).contiguous().view(b_size*t_size, c_size, d_size)
		
		if key_word is not None:
			key_sent = key_word.view(b_size, c_size, t_size, d_size).transpose(1,2).contiguous().view(b_size*t_size, c_size, d_size)
		else:
			 key_sent = context_sent

		"""  Create the padding mask for context  """  
		mask_sent = index < 0
		context_pad_mask = mask_sent.unsqueeze(1).expand(b_size, 
					t_size, c_size).contiguous().view(b_size*t_size, -1)
		context_pad_mask = context_pad_mask.unsqueeze(1).contiguous()

		"""  Reshape query  """ 
		query_ = query.view(b_size*t_size, 1, d_size)
		
		_,h,t,s = attn_word.size()
		aeq(t, t_size)
		attn_word = attn_word.view(b_size, c_size, h, t, s)

		return query_, context_sent, key_sent, context_pad_mask, attn_word




	"""
	Function to obtein index for selectin contextual sentences

	Args:
	  doc_index:List of indexes where a document starts in the batch 				
	  b_size: 	Query batch size 
	  b_size_:	Context batch size
	  batch_i:	None while training. While generating translation indicates the current example in the batch

	Out:
	  index:	(LongTensor) Matrix that indicates the index of the context for each example in the query 
					{batch_1 x context_size}
				For example: Having batch_1=2 and context_size=3, index=[[0,1,2], [1,2,3]] . 
							 Here we take the sentence 0, 1 and 2 from context for the first query, and 
							 1, 2 and 3 from context for the second query.
	"""
	def get_context_index(self, doc_index, b_size, b_size_, batch_i):
		
		if batch_i is None:
			index = np.matrix([range(b_size)])-1
			index = index.repeat(self.context_size, axis=0)
			mask = np.zeros([self.context_size, self.context_size], dtype=bool)

			for i in range(self.context_size):
				index[i,:] -= self.context_size-1-i
				mask[i,0:self.context_size-i] = True

			for i in doc_index:
				index[:,i:min(i+self.context_size, b_size)][mask[:,:min(self.context_size, b_size - i)]] = -1
			index = np.transpose(index)
		else:
			if batch_i == 0:
				index = np.zeros([b_size, self.context_size])-1
			else:
				assert b_size_ <= self.context_size, "Error in context"
				index = np.matrix(range(b_size_-self.context_size, 0) + range(b_size_))		
				index = index.repeat(b_size, axis=0)	

		return index




	"""
	Forward function 

	Args:
	  input:	(LongTensor) Encoder/decoder input 
					{sentence_lenght x batch_1 x 1} 				
	  query: 	(FloatTensor) Encoder/decoder hidden states of current sentence 
					{sentence_lenght_1 x batch_1 x  dim_hidden_state}
	  context: 	(FloatTensor) Encoder/decoder hidden states of previous sentences 
					{sentence_lenght_2 x batch_2 x dim_hidden_state}
	  doc_index:List of indexes where a document starts in the batch
	  batch_i:	None while training. While generating translation indicates the current example in the batch 

	  NOTE: Query and context are usually the same tensor during training. We divided into 2 variables for generalization.

	Out:
	  out:		(FloatTensor) Contextualized encoder/decoder hidden states of current sentence 
					{sentence_lenght_1 x batch_1 x  dim_hidden_state}
	  attn_word:(FloatTensor) Word-level attention weights 
					 {batch_1 x context_size x heads x sentence_lenght_1 x sentence_lenght_2}
	  attn_sent:(FloatTensor) Sentence-level attention weights 
					 {batch_1 x heads x 1 x context_size}
	"""

	def forward(self, input, query, context, doc_index, batch_i=None):

		input = input[:, :, 0].transpose(0,1).contiguous() 
		query = query.transpose(0,1).contiguous()
		context = context.transpose(0,1).contiguous()

		"""   Create word padding mask   """
		in_batch, in_len = input.size() 
		in_pad_mask = input.data.eq(self.padding_idx)
		in_pad_mask = in_pad_mask.unsqueeze(1).expand(in_batch, query.size()[1], in_len)

		b_size, t_size, d_size = query.size()
		b_size_, k_size, d_size = context.size()

		"""   Transform query for word and sentence   """
		query_word_norm = self.layer_norm_query_word(query)
		query_sent_norm = self.layer_norm_query_sent(query)
	
		index = torch.Tensor(self.get_context_index(doc_index, b_size, b_size_, batch_i)).type_as(query.data).long()
		
		"""   Reshape the tensors for appliying word-level attention   """   
		query_word_norm, context_word, key_word, context_word_mask = self._get_word_context(query_word_norm, context, key, index, in_pad_mask)

		"""   Word-level attention   """   
		context_word, attn_word = self.word_attn(key_word, context_word,
										query_word_norm, mask=context_word_mask, return_key=key is not None, all_attn=True)
		context_sent = self.layer_norm_word(context_word)

		"""   Reshape the tensors for appliying sentence-level attention   """ 
		query_sent_norm, context_sent, key_sent, context_sent_mask, attn_word = self._get_sent_context(query_sent_norm, 
																context_sent, None if key is None else key_word, index, attn_word)

		"""   Sentence-level attention   """
		sent_context, attn_sent = self.sent_attn(key_sent, context_sent, query_sent_norm,
									  mask=context_sent_mask, all_attn=True)	
		
		sent_context = sent_context.view(b_size, t_size, d_size)
		sent_context = self.feed_forward(sent_context)

		"""   Calculate gate   """
		l = self.sigmoid(self.linear(torch.cat([query, sent_context], dim=2)))
		out = (1-l)*query + l*sent_context

		out = self.layer_norm_sent(out)
		
		return out.transpose(0,1).contiguous(), attn_word, attn_sent
