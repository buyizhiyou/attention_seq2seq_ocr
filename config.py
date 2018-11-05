#-*-coding:utf8-*-

__author = "buyizhiyou"
__date = "2018-10-30"

from easydict import EasyDict as edict 

config = edict()
_C = config

_C.enc_lstm_dim=128
_C.dec_lstm_dim=256
_C.num_units=512
_C.embed_dim=100
_C.batch_size=32
_C.vocab_size=70
_C.learning_rate=0.0001
_C.input_max_length=100
_C.output_max_length=30
_C.epoches = 100


_C.checkpoint_path = "checkpoint/"
_C.log_path = "logs/"
_C.pretrained_model = "vgg_pretrained/VGG_imagenet.npy"


