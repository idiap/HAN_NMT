import argparse
import torch
import codecs
import os
import math
import numpy
import sys

from torch.autograd import Variable
from itertools import count

import onmt.ModelConstructor
import onmt.translate.Beam
import onmt.io
import onmt.opts
import time

def make_translator(opt, report_score=True, out_file=None):
        if out_file is None:
                out_file = codecs.open(opt.output, 'w', 'utf-8')
        out_attn = opt.output + ".attn"
        if opt.gpu > -1:
                torch.cuda.set_device(opt.gpu)

        dummy_parser = argparse.ArgumentParser(description='train.py')
        onmt.opts.model_opts(dummy_parser)
        dummy_opt = dummy_parser.parse_known_args([])[0]

        fields, model, model_opt = \
                onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

        scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                                                                         opt.beta,
                                                                                         opt.coverage_penalty,
                                                                                         opt.length_penalty)

        kwargs = {k: getattr(opt, k)
                          for k in ["data_type", "gpu", "verbose"]}

        translator = Translator(model, fields, beam_size=opt.beam_size, global_scorer=scorer,
                                                        out_file=out_file, report_score=report_score,
                                                        copy_attn=model_opt.copy_attn, context_size=model_opt.context_size, 
                                                    context_type=model_opt.context_type, out_attn=out_attn, **kwargs)
        return translator

class Translator(object):
	"""
	Uses a model to translate a batch of sentences.


	Args:
	   model (:obj:`onmt.modules.NMTModel`):
		  NMT model to use for translation
	   fields (dict of Fields): data fields
	   beam_size (int): size of beam to use
	   n_best (int): number of translations produced
	   max_length (int): maximum length output to produce
	   global_scores (:obj:`GlobalScorer`):
		 object to rescore final translations
	   copy_attn (bool): use copy attention during translation
	   cuda (bool): use cuda
	   beam_trace (bool): trace beam search for debugging
	"""

	def __init__(self,
				 model,
				 fields,
				 beam_size=5,
				 n_best=1,
				 max_length=100,
				 global_scorer=None,
				 copy_attn=False,
				 context_size=3,
				 context_type="",
				 gpu=False,
				 dump_beam="",
				 min_length=0,
				 stepwise_penalty=False,
				 block_ngram_repeat=0,
				 ignore_when_blocking=[],
				 sample_rate='16000',
				 window_size=.02,
				 window_stride=.01,
				 window='hamming',
				 use_filter_pred=False,
				 data_type="text",
				 replace_unk=False,
				 report_score=True,
				 report_bleu=False,
				 report_rouge=False,
				 verbose=False,
				 out_file=None,
				 translate_part="all",
				 out_attn=None):
		self.gpu = gpu
		self.cuda = gpu > -1

		self.model = model
		self.fields = fields
		self.n_best = n_best
		self.max_length = max_length
		self.global_scorer = global_scorer
		self.copy_attn = copy_attn
		self.beam_size = beam_size
		self.min_length = min_length
		self.stepwise_penalty = stepwise_penalty
		self.dump_beam = dump_beam
		self.block_ngram_repeat = block_ngram_repeat
		self.ignore_when_blocking = set(ignore_when_blocking)
		self.sample_rate = sample_rate
		self.window_size = window_size
		self.window_stride = window_stride
		self.window = window
		self.use_filter_pred = use_filter_pred
		self.replace_unk = replace_unk
		self.data_type = data_type
		self.verbose = verbose
		self.out_file = out_file
		self.report_score = report_score
		self.report_bleu = report_bleu
		self.report_rouge = report_rouge
		self.translate_part = translate_part
		self.context_size = context_size
		self.context_type = context_type
		self.out_attn = out_attn

		# for debugging
		self.beam_trace = self.dump_beam != ""
		self.beam_accum = None
		if self.beam_trace:
			self.beam_accum = {
				"predicted_ids": [],
				"beam_parent_ids": [],
				"scores": [],
				"log_probs": []}

	def translate(self, src_dir, src_path, tgt_path, doc_path,
				  batch_size, attn_debug=False):
		data = onmt.io.build_dataset(self.fields,
									 self.data_type,
									 src_path,
									 tgt_path,
									 doc_path=doc_path,
									 src_dir=src_dir,
									 sample_rate=self.sample_rate,
									 window_size=self.window_size,
									 window_stride=self.window_stride,
									 window=self.window,
									 use_filter_pred=self.use_filter_pred)

		data_iter = onmt.io.DocumentIterator(
			dataset=data, device=self.gpu,
			batch_size=batch_size, train=False,
			sort_within_batch=False, shuffle=False)

		builder = onmt.translate.TranslationBuilder(
			data, self.fields,
			self.n_best, self.replace_unk, tgt_path)

		# Statistics
		counter = count(1)
		pred_score_total, pred_words_total = 0, 0
		gold_score_total, gold_words_total = 0, 0

		all_scores = []
		ctx_attn = []
		start_time = time.time()
		for batch, context in data_iter:
			batch_data = self.translate_batch(batch, data, context, self.translate_part)
			translations = builder.from_batch(batch_data)

			for trans in translations:
				all_scores += [trans.pred_scores[0]]
				pred_score_total += trans.pred_scores[0]
				pred_words_total += len(trans.pred_sents[0])
				if tgt_path is not None:
					gold_score_total += trans.gold_score
					gold_words_total += len(trans.gold_sent) + 1
					ctx_attn.append(trans.ctx_attn)

				n_best_preds = [" ".join(pred)
								for pred in trans.pred_sents[:self.n_best]]
				self.out_file.write('\n'.join(n_best_preds) + '\n')
				self.out_file.flush()

				if self.verbose:
					sent_number = next(counter)
					output = trans.log(sent_number)
					os.write(1, output.encode('utf-8'))

				# Debug attention.
				if attn_debug:
					srcs = trans.src_raw
					preds = trans.pred_sents[0]
					preds.append('</s>')
					attns = trans.attns[0].tolist()
					header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
					row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
					output = header_format.format("", *trans.src_raw) + '\n'
					for word, row in zip(preds, attns):
						max_index = row.index(max(row))
						row_format = row_format.replace(
							"{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
						row_format = row_format.replace(
							"{:*>10.7f} ", "{:>10.7f} ", max_index)
						output += row_format.format(word, *row) + '\n'
						row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
					os.write(1, output.encode('utf-8'))

		end_time = time.time() - start_time
		if self.report_score:
			self._report_score('PRED', pred_score_total, pred_words_total)
			if tgt_path is not None:
				self._report_score('GOLD', gold_score_total, gold_words_total)
				if self.report_bleu:
					self._report_bleu(tgt_path)
				if self.report_rouge:
					self._report_rouge(tgt_path)
		print pred_words_total, end_time, pred_words_total/end_time
		if self.dump_beam:
			import json
			json.dump(self.translator.beam_accum,
					  codecs.open(self.dump_beam, 'w', 'utf-8'))
		if len(ctx_attn) > 0:
			torch.save(ctx_attn, self.out_attn)
		return all_scores

	def translate_batch(self, batch, data, context, translate_part):
		"""
		Translate a batch of sentences.

		Mostly a wrapper around :obj:`Beam`.

		Args:
		   batch (:obj:`Batch`): a batch from a dataset object
		   data (:obj:`Dataset`): the dataset object


		Todo:
		   Shouldn't need the original dataset.
		"""

		# (0) Prep each of the components of the search.
		# And helper method for reducing verbosity.
		beam_size = self.beam_size
		batch_size = batch.batch_size
		data_type = data.data_type
		vocab = self.fields["tgt"].vocab

		# Define a list of tokens to exclude from ngram-blocking
		# exclusion_list = ["<t>", "</t>", "."]
		exclusion_tokens = set([vocab.stoi[t]
								for t in self.ignore_when_blocking])

		beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
									cuda=self.cuda,
									global_scorer=self.global_scorer,
									pad=vocab.stoi[onmt.io.PAD_WORD],
									eos=vocab.stoi[onmt.io.EOS_WORD],
									bos=vocab.stoi[onmt.io.BOS_WORD],
									min_length=self.min_length,
									stepwise_penalty=self.stepwise_penalty,
									block_ngram_repeat=self.block_ngram_repeat,
									exclusion_tokens=exclusion_tokens)
				for __ in range(batch_size)]

		# Help functions for working with beams and batches
		def var(a): return Variable(a, volatile=True)

		def rvar(a): return var(a.repeat(1, beam_size, 1))

		def bottle(m):
			return m.view(batch_size * beam_size, -1)

		def unbottle(m):
			return m.view(beam_size, batch_size, -1)

		# (1) Run the encoder on the src.
		src = onmt.io.make_features(batch, 'src', data_type)
		src_lengths = None
		if data_type == 'text':
			_, src_lengths = batch.src

		enc_states, memory_bank = self.model.encoder(src, src_lengths)
		
		if self.context_type and "HAN_join" in self.context_type:
			memory_bank, _, _ = self.model.doc_context[0](src, memory_bank, memory_bank, context)
		elif self.context_type in {"HAN_enc", "HAN_dec_source"}:
			memory_bank, _, _ = self.model.doc_context(src, memory_bank, memory_bank, context)

		if src_lengths is None:
			src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
												  .long()\
												  .fill_(memory_bank.size(0))


		ret = {"predictions": [],
			   "scores": [],
			   "attention": []}

		cache, ind_cache = [src[:,0:1,:], enc_states[:,0:1,:], enc_states[:,0:1,:]], []


		for batch_i in range(batch_size):

			if isinstance(enc_states, tuple):
				enc_states_i = (enc_states[0][:,batch_i:batch_i+1,:], enc_states[1][:,batch_i:batch_i+1,:])
			else:
				enc_states_i = enc_states[:,batch_i:batch_i+1,:]

			dec_states = self.model.decoder.init_decoder_state(
										src[:,batch_i:batch_i+1,:], memory_bank[:,batch_i:batch_i+1,:], enc_states_i)

			# (2) Repeat src objects `beam_size` times.
			src_map = batch.src_map[batch_i:batch_i+1].data \
				 if data_type == 'text' and self.copy_attn else None
			memory_bank_i = rvar(memory_bank[:, batch_i:batch_i+1, :].data)
			memory_lengths = src_lengths[batch_i:batch_i+1].repeat(beam_size)
			dec_states.repeat_beam_size_times(beam_size)

			# (3) run the decoder to generate sentences, using beam search.
			for i in range(self.max_length):
				if beam[batch_i].done():
					break

				# Construct batch x beam_size nxt words.
				# Get all the pending current beam words and arrange for forward.
				inp = var(beam[batch_i].get_current_state().view(1, -1))

				# Turn any copied words to UNKs
				# 0 is unk
				if self.copy_attn:
					inp = inp.masked_fill(
						inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

				# Temporary kludge solution to handle changed dim expectation
				# in the decoder
				inp = inp.unsqueeze(2)

				# Run one step.
				dec_out, dec_states, attn, mid = self.model.decoder(
					inp, memory_bank_i, dec_states, memory_lengths=memory_lengths)
				
				# dec_out: beam x rnn_size

				if self.model.doc_context and translate_part in ["all", "context"]:
					if self.context_type == "HAN_join":
						dec_out, _,_ = self.model.doc_context[1](cache[0], dec_out, cache[1], context, batch_i=batch_i)
					elif "HAN_dec" in self.context_type: 
						dec_out, _,_  = self.model.doc_context(cache[0], dec_out, cache[1], context, batch_i=batch_i)
				
				dec_out = dec_out.squeeze(0)

				# (b) Compute a vector of batch x beam word scores.
				if not self.copy_attn:
					out = self.model.generator.forward(dec_out).data
					out = out.view(beam_size, 1, -1)
					# beam x tgt_vocab
					beam_attn = attn["std"].view(beam_size, 1, -1)
				else:
					out = self.model.generator.forward(dec_out,
						attn["copy"].squeeze(0), src_map)
					# beam x (tgt_vocab + extra_vocab)
					out = data.collapse_copy_scores(
						out.view(beam_size, 1, -1),
						batch[batch_i], self.fields["tgt"].vocab, data.src_vocabs)
					# beam x tgt_vocab
					out = out.log()
					beam_attn = attn["copy"].view(beam_size, 1, -1)
				# (c) Advance each beam.
				beam[batch_i].advance(out[:, 0], beam_attn.data[:, 0, :memory_lengths[0]])
				dec_states.beam_update(0, beam[batch_i].get_current_origin(), beam_size)

			self._from_beam(beam[batch_i], ret)
			cache, ind_cache = self.update_context(ret["predictions"][-1][0], cache, ind_cache,  
									enc_states, src, memory_bank, src_lengths, batch_i, translate_part, vocab.stoi[onmt.io.PAD_WORD])
		
		# (4) Extract sentences from beam.
		#ret = self._from_beam(beam)
		ret["gold_score"] = [0] * batch_size
		ret["ctx_attn"] = None
		if "tgt" in batch.__dict__:
			ret["gold_score"], ret["ctx_attn"] = self._run_target(batch, data, context, translate_part)
		ret["batch"] = batch
		print ret
		return ret

	def update_context(self, pred, cache, ind_cache, enc_states, src, memory_bank, src_lengths, batch_i, translate_part, pad):
		
		if self.context_type == "HAN_enc" or self.context_type == "HAN_dec_source":
			b_len = min(self.context_size-1, batch_i)
			cache = [src[:,batch_i-b_len:batch_i+1,:], memory_bank[:,batch_i-b_len:batch_i+1,:]]

		elif self.context_type in {"HAN_dec", "HAN_dec_context", "HAN_join"}:
			ind_cache.append(pred)
			if len(ind_cache) > self.context_size:
				del ind_cache[0]
			s_len = max([len(i) for i in ind_cache])
			b_len = len(ind_cache)
			pred = numpy.empty([s_len, b_len])
			pred.fill(pad)
			for i in range(b_len):
				pred[:len(ind_cache[i]),i] = ind_cache[i]
			
			prev_context = Variable(torch.Tensor(pred).type_as(memory_bank.data).long().unsqueeze(2))
			prev_context = prev_context[:-1]
			prev_out, prev_memory_bank = self.run_decoder(prev_context, enc_states, src, memory_bank, src_lengths, batch_i, b_len-1)
			
			if self.context_type == "HAN_dec_context":
				cache = [prev_context, prev_memory_bank, None]
			else:
				cache = [prev_context, prev_out, None]

		return cache, ind_cache
		

	def run_decoder(self, prev_context, enc_states, src, memory_bank, src_lengths, batch_i, b_len=0):		
		if isinstance(enc_states, tuple):
			enc_states_i = (enc_states[0][:,batch_i-b_len:batch_i+1,:], enc_states[1][:,batch_i-b_len:batch_i+1,:])
		else:
			enc_states_i = enc_states[:,batch_i-b_len:batch_i+1,:]

		dec_states = self.model.decoder.init_decoder_state(src[:,batch_i-b_len:batch_i+1,:], 
							memory_bank[:,batch_i-b_len:batch_i+1,:], 
							enc_states_i)
		out, _, _, mid = self.model.decoder(prev_context, 
							memory_bank[:,batch_i-b_len:batch_i+1,:], 
							dec_states, 
							memory_lengths=src_lengths[batch_i-b_len:batch_i+1])
		
		return out, mid 

	def _from_beam(self, b, ret):
		n_best = self.n_best
		scores, ks = b.sort_finished(minimum=n_best)
		hyps, attn = [], []
		for i, (times, k) in enumerate(ks[:n_best]):
			hyp, att = b.get_hyp(times, k)
			hyps.append(hyp)
			attn.append(att)
		ret["predictions"].append(hyps)
		ret["scores"].append(scores)
		ret["attention"].append(attn)


	def _run_target(self, batch, data, context, translate_part):
		data_type = data.data_type
		if data_type == 'text':
			_, src_lengths = batch.src
		else:
			src_lengths = None
		src = onmt.io.make_features(batch, 'src', data_type)
		tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

		#  (1) run the encoder on the src
		enc_states, memory_bank = self.model.encoder(src, src_lengths)
		attn_word_enc, attn_sent_enc, attn_word_dec, attn_sent_dec = None, None, None, None


		if self.model.doc_context and translate_part in ["all", "context"]:
			if self.context_type == "HAN_join":
				memory_bank, attn_word_enc, attn_sent_enc = self.model.doc_context[0](src, memory_bank, memory_bank, context)
			elif self.context_type == "HAN_enc" or self.context_type == "HAN_dec_source":
				memory_bank, attn_word_enc, attn_sent_enc = self.model.doc_context(src, memory_bank, memory_bank, context)

		dec_states = \
			self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

		#  (2) if a target is specified, compute the 'goldScore'
		#  (i.e. log likelihood) of the target under the model
		tt = torch.cuda if self.cuda else torch
		gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
		dec_out, _, attns, mid = self.model.decoder(
			tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

		if self.model.doc_context and translate_part in ["all", "context"]:
			if self.context_type == "HAN_join":
				decoder_outputs, attn_word_dec, attn_sent_dec = self.model.doc_context[1](tgt_in, dec_out, dec_out, context)
			elif "HAN_dec" in self.context_type:
				if self.context_type == "HAN_dec_source":
					ctxt = memory_bank
					inp = src
				if self.context_type == "HAN_dec_context":
					ctxt = mid
					inp = tgt_in
				else: 
					ctxt = dec_out
					inp = tgt_in

				dec_out, attn_word_dec, attn_sent_dec = self.model.doc_context(inp, query, ctxt, context)

		tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
		for dec, tgt in zip(dec_out, batch.tgt[1:].data):
			# Log prob of each word.
			out = self.model.generator.forward(dec)
			tgt = tgt.unsqueeze(1)
			scores = out.data.gather(1, tgt)
			scores.masked_fill_(tgt.eq(tgt_pad), 0)
			gold_scores += scores
		return gold_scores

	def _report_score(self, name, score_total, words_total):
		print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
			name, score_total / words_total,
			name, math.exp(-score_total / words_total)))

	def _report_bleu(self, tgt_path):
		import subprocess
		path = os.path.split(os.path.realpath(__file__))[0]
		print()

		res = subprocess.check_output("perl %s/tools/multi-bleu.perl %s"
									  % (path, tgt_path, self.output),
									  stdin=self.out_file,
									  shell=True).decode("utf-8")

		print(">> " + res.strip())

	def _report_rouge(self, tgt_path):
		import subprocess
		path = os.path.split(os.path.realpath(__file__))[0]
		res = subprocess.check_output(
			"python %s/tools/test_rouge.py -r %s -c STDIN"
			% (path, tgt_path),
			shell=True,
			stdin=self.out_file).decode("utf-8")
		print(res.strip())
