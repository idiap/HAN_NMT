## Description

Implementation of the paper ["Document-Level Neural Machine Translation with Hierarchical Attention Networks"](http://www.aclweb.org/anthology/D18-1325). It is based on OpenNMT (v.2.1) https://github.com/OpenNMT/OpenNMT-py

This is a restricted version. It DOES NOT work for shards, and multimodal translation.

## Preprocess
The data, similary for any NMT baseline, consists of a source file and a target file which are aligned at sentence-level. However, the sentences should be in order for each document (i.e. not shuffled). Additionally, the model requires a file (doc_file) indicating the beginning of each document in the source file. Each line of the doc_file indicates the number of lines at the source file where a new document starts. 

Example: 

>	0  
>	10  
>	25 

There are 3 documents. The first one from line 0 to line 9, the second from line 10 to 24, the third from line 25 to the end.

Command:
```
python preprocess.py -train_src [source_file] -train_tgt [target_file] -train_doc [doc_file] 
-valid_src [source_dev_file] -valid_tgt [target_dev_file] -valid_doc [doc_dev_file] -save_data [out_file]
```
The folder preprocess_TED_zh-en contains the files to preprocess the TED Talks zh-en dataset from https://wit3.fbk.eu/mt.php?release=2015-01.

## Training
Training the sentence-level NMT baseline:

```
python train.py -data [data_set] -save_model [sentence_level_model] -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 4096 -start_decay_at 20 -report_every 500 -epochs 20 -gpuid 0 -max_generator_batches 16 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot 
-train_part sentences
```

Training HAN-encoder using the sentence-level NMT model:

```
python train.py -data [data_set] -save_model [HAN_enc_model] -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 1024 -start_decay_at 2 -report_every 500 -epochs 1 -gpuid 0 -max_generator_batches 32 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot 
-train_part all -context_type HAN_enc -context_size 3 -train_from [sentence_level_model]
```

Training HAN-decoder using the sentence-level NMT model:

```
python train.py -data [data_set] -save_model [HAN_dec_model] -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 1024 -start_decay_at 2 -report_every 500 -epochs 1 -gpuid 0 -max_generator_batches 32 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot 
-train_part all -context_type HAN_dec -context_size 3 -train_from [sentence_level_model]
```

Training HAN-joint using the HAN-encoder model:

```
python train.py -data [data_set] -save_model [HAN_joint_model] -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 1024 -start_decay_at 2 -report_every 500 -epochs 1 -gpuid 0 -max_generator_batches 32 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot 
-train_part all -context_type HAN_join -context_size 3 -train_from [HAN_enc_model]
```

Input options:

- train_part:	[sentences, context, all]   
- context_type:	[HAN_enc, HAN_dec, HAN_join, HAN_dec_source, HAN_dec_context]
- context_size:	number of previous sentences

NOTE: The transformer model is sensitive to variation on hyperparameters. The HAN is also sensitive to the batch size.

## Translation
The translation is done sentence by sentence despite not being necesary for HAN_enc or baseline (this could be improved).

```
python translate.py -model [model] -src [test_source_file] -doc [test_doc_file] 
-output [out_file] -translate_part all -batch_size 1000 -gpu 0
```
Input options:

- translate_part: [sentences, all]
- batch_size: maximun number of sentences to keep in memory at once.


## Test files reported in the paper
The output files of the 3 reported systems: transformer NMT, cache NMT, HAN-decoder NMT, HAN-encoder NMT, HAN-encoder-decoder NMT.
>	- sub_es-en: Opensubtitles 
>	- sub_zh-en: TV subtitles 
>	- TED_es-en: TED Talks WIT 2015
>	- TED_zh-en: TED Talks WIT 2014


## Reference:
>Miculicich, L., Ram, D., Pappas, N. & Henderson, J. Document-Level Neural Machine Translation with Hierarchical Attention Networks. EMNLP 2018.

```
@INPROCEEDINGS{Miculicich_EMNLP_2018,
         author = {Miculicich, Lesly and Ram, Dhananjay and Pappas, Nikolaos and Henderson, James},
       projects = {Idiap, SUMMA},
          title = {Document-Level Neural Machine Translation with Hierarchical Attention Networks},
      booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
           year = {2018},
            pdf = {http://publications.idiap.ch/downloads/papers/2018/Miculicich_EMNLP_2018.pdf}
}
```
 
## Contact:
lmiculicich@idiap.ch
