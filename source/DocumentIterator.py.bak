### Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
### Written by Lesly Miculicich <lmiculicich@idiap.ch>

import os, sys
from collections import Counter, defaultdict, OrderedDict
from itertools import count

import torch
import torchtext.data
import torchtext.vocab
import numpy as np



class DocumentIterator(torchtext.data.Iterator):
	

	def __init__(self, dataset, batch_size, device=None,
				 batch_size_fn=None, train=True, shuffle=None,
				 sort_within_batch=None):
		
		super(DocumentIterator, self).__init__(dataset, batch_size, device=device,
				 batch_size_fn=batch_size_fn, train=train,
				 repeat=False, shuffle=False, sort=False,
				 sort_within_batch=sort_within_batch)
		self.doc_index, self.doc_range = self.get_context_index(self.data())
	
		self.indx = None

	def document_shuffler(self):
		shuffler_index = self.random_shuffler(range(len(self.doc_range)))
		docs, indx = [], []
		for i in shuffler_index:
			docs.extend(self.dataset[self.doc_range[i][0]:self.doc_range[i][1]])
			indx.extend(self.doc_index[self.doc_range[i][0]:self.doc_range[i][1]])

		assert len(docs) == len(self.doc_index), "Error in document indexes"
		assert len(indx) == len(self.dataset), "Error in document indexes"

		return docs, np.array(indx)

	def create_batches(self):
		if self.train:
			data, indx = self.document_shuffler()
			self.batches = torchtext.data.batch(data, self.batch_size, self.batch_size_fn)
			self.indx = indx
		else:
			self.batches = self.batch_eval()
			self.indx = np.array(self.doc_index)


	def get_context_index(self, batch):
		d_index, d_range, prev_i, i = [False]*len(batch), [], 0, 0
		for i, m in enumerate(batch):
			if m.indices in self.dataset.doc_index:
				d_index[i] = True
				if prev_i != i: d_range.append((prev_i, i))
				prev_i = i
		if prev_i != i+1: d_range.append((prev_i, i+1))			
		return d_index, d_range

	def __iter__(self):
		while True:
			self.init_epoch()
			count = 0
			for idx, minibatch in enumerate(self.batches):
				# fast-forward if loaded from state
				if self._iterations_this_epoch > idx:
					continue
				self.iterations += 1
				self._iterations_this_epoch += 1
				indx = np.where(self.indx[count:count + len(minibatch)])[0].tolist()
				count += len(minibatch)
				yield torchtext.data.Batch(minibatch, self.dataset, self.device, self.train), indx
			if not self.repeat:
				raise StopIteration

	def batch_eval(self):	
		for r in self.doc_range:
			if r[1]-r[0] > self.batch_size:
				for i in range(int((r[1]-r[0])/self.batch_size)):
					yield self.dataset[r[0]+i*self.batch_size:r[0]+i*self.batch_size+self.batch_size]
				if r[0]+i*self.batch_size+self.batch_size < r[1]:
					yield self.dataset[r[0]+i*self.batch_size+self.batch_size:r[1]]
			else:
				yield self.dataset[r[0]:r[1]]
	
