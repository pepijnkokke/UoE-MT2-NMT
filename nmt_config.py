# coding: utf-8
#---------------------------------------------------------------------
'''
Neural Machine Translation - Configuration file
        used to set parameters for training models
'''
#---------------------------------------------------------------------
import os
#---------------------------------------------------------------------
# Internal Parameters - DO NOT MODIFY -- (ง'-')ง
#---------------------------------------------------------------------
max_vocab_size = {"en" : 10000, "fr" : 10000}

# Special vocabulary symbols
PAD = b"_PAD"
GO = b"_GO"
EOS = b"_EOS"
UNK = b"_UNK"
START_VOCAB = [PAD, GO, EOS, UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Attention flags
NO_ATTN = 0
SOFT_ATTN = 1
HARD_ATTN = 2

# for appending post fix to output
attn_post = ["NO_ATTN", "SOFT_ATTN", "HARD_ATTN"]

# maximum available sentences in dataset
NUM_SENTENCES = 10500
FREQ_THRESH = 1

'''
Input directory contains
- parallel japanese-english corpus
- word to integer mappings (and reverse)
- vocabulary dictionary
'''
input_dir = os.path.join("data")

# model directory is used to store trained models and log files
model_dir = os.path.join("model")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(input_dir):
    print("Input folder not found".format(input_dir))

text_fname = {"en": os.path.join(input_dir, "text.en"), "fr": os.path.join(input_dir, "text.fr")}
tokens_fname = os.path.join(input_dir, "tokens.list")
vocab_path = os.path.join(input_dir, "vocab.dict")
w2i_path = os.path.join(input_dir, "w2i.dict")
i2w_path = os.path.join(input_dir, "i2w.dict")

#---------------------------Select options below------------------

#-----------------------------------------------------------------
# Japanese English (from Lamtran) configuration
#-----------------------------------------------------------------
print("Japanese English dataset configuration")

# Set experiment name - can be used to save models with different
# names
EXP_NAME = "ja_en_exp1"

# the maximum number of words to predict if EOS not predicted
MAX_PREDICT_LEN = 20

# change flag to use reduced dataset, 1000 datasets
USE_ALL_DATA = True

if USE_ALL_DATA:
  # number of training examples to use
  NUM_TRAINING_SENTENCES = NUM_SENTENCES-500
  # number of validation examples to use
  NUM_DEV_SENTENCES = 500
else:
  # number of training examples to use
  NUM_TRAINING_SENTENCES = 1000
  # number of validation examples to use
  NUM_DEV_SENTENCES = 100
#---------------------------------------------------------------------
# Model Parameters
#---------------------------------------------------------------------
# number of LSTM layers for encoder
num_layers_enc = 1
# number of LSTM layers for decoder
num_layers_dec = 1
# number of hidden units per LSTM
# both encoder, decoder are similarly structured
hidden_units = 100
# default model - no attention
# when implementing attention use either - SOFT_ATTN or HARD_ATTN
use_attn = NO_ATTN
'''
KEEP this flag true to avoid losing earlier trained models
The code checks if a trained model file with the selected parameters
exists. If it does, it needs to be manually deleted by the user.
'''
load_existing_model = True
#---------------------------------------------------------------------
# Training Parameters
#---------------------------------------------------------------------
# Training EPOCHS
NUM_EPOCHS = 0
# if >= 0, use GPU, if negative use CPU
gpuid = -1
#---------------------------------------------------------------------
# Log file details - changing the following names not recommended
#---------------------------------------------------------------------
name_to_log = "{0:d}sen_{1:d}-{2:d}layers_{3:d}units_{4:s}_{5:s}".format(
                                                            NUM_TRAINING_SENTENCES,
                                                            num_layers_enc,
                                                            num_layers_dec,
                                                            hidden_units,
                                                            EXP_NAME,
                                                            attn_post[use_attn])

log_train_fil_name = os.path.join(model_dir, "train_{0:s}.log".format(name_to_log))
model_fil = os.path.join(model_dir, "seq2seq_{0:s}.model".format(name_to_log))
#---------------------------------------------------------------------
