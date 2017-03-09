# coding: utf-8
#---------------------------------------------------------------------
'''
Neural Machine Translation - Encoder Decoder model
    Chainer implementation of an encoder-decoder sequence to sequence
    model using bi-directional LSTM encoder
'''
#---------------------------------------------------------------------

# In[]:

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.functions.array.concat as A
import chainer.functions.noise.dropout as N
import chainer.links as L
from chainer.training import extensions
from chainer.functions.array import concat


# In[]:
# Import configuration file

from nmt_config import *

# In[]:

class EncoderDecoder(Chain):

    '''
    Constructor to initialize model
    Params:
        vsize_enc   - vocabulary size for source language (fed into encoder)
        vsize_dec   - vocabulary size for target language (fed into decoder)
        n_units     - size of the LSTMs
        attn        - specifies whether to use attention
    '''
    def __init__(self, vsize_enc, vsize_dec,
                 nlayers_enc, nlayers_dec,
                 n_units, gpuid, attn=False):
        super(EncoderDecoder, self).__init__()
        #--------------------------------------------------------------------
        # add encoder layers
        #--------------------------------------------------------------------
        # add embedding layer
        self.add_link("embed_enc", L.EmbedID(vsize_enc, n_units))

        '''
        ___QUESTION-1-DESCRIBE-A-START___
        Create `nlayers_enc` encoding layers using add_link(). There are two
        loops for adding layers because we create one encoder to pass over the
        sentence in order, and one to pass over the sentence in reverse order.
        '''
        self.lstm_enc = ["L{0:d}_enc".format(i) for i in range(nlayers_enc)]
        for lstm_name in self.lstm_enc:
            self.add_link(lstm_name, L.LSTM(n_units, n_units))

        self.lstm_rev_enc = ["L{0:d}_rev_enc".format(i) for i in range(nlayers_enc)]
        for lstm_name in self.lstm_rev_enc:
            self.add_link(lstm_name, L.LSTM(n_units, n_units))
        '''
        ___QUESTION-1-DESCRIBE-A-END___
        '''

        #--------------------------------------------------------------------
        # add decoder layers
        #--------------------------------------------------------------------
        # add embedding layer
        '''
        ___QUESTION-1-DESCRIBE-B-START___
        Comment on the input and output sizes of the following layers:
        - L.EmbedID(vsize_dec, 2*n_units)
        - L.LSTM(2*n_units, 2*n_units)
        - L.Linear(2*n_units, vsize_dec)
        Why are we using multipliers over the base number of units (n_units)?

        The encoding of a sentence is a matrix, each column of which is the
        concatenation of the hidden states of the forward and the backward
        encoder LSTMs at that position. Since both the LSTMs are sized n_units,
        the concatenation of these two hidden states is 2*n_units.
        '''

        self.add_link("embed_dec", L.EmbedID(vsize_dec, 2*n_units))

        # add LSTM layers
        self.lstm_dec = ["L{0:d}_dec".format(i) for i in range(nlayers_dec)]
        for lstm_name in self.lstm_dec:
            self.add_link(lstm_name, L.LSTM(2*n_units, 2*n_units))

        if attn > 0:
            # __QUESTION Add attention
            self.add_link("attn_out", L.Linear(4*n_units, 2*n_units))

        # Save the attention preference
        # __QUESTION you should use this flag to check if attention
        # has been selected. Your code should work with and without attention
        self.attn = attn

        # add output layer
        self.add_link("out", L.Linear(2*n_units, vsize_dec))
        '''
        ___QUESTION-1-DESCRIBE-B-END___
        '''

        # Store GPU id
        self.gpuid = gpuid
        self.n_units = n_units

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.lstm_enc + self.lstm_rev_enc + self.lstm_dec:
            self[lstm_name].reset_state()
        self.loss = 0

    '''
    ___QUESTION-1-DESCRIBE-C-START___
    Describe what the function set_decoder_state() is doing. What are c_state
    and h_state?

    The method set_decoder_state() will feed the encoder output into the
    decoder. The initial implementation takes the cell states and hidden state
    from the final LSTM of the encoder, and feeds them into the first LSTM of
    the decoder.
    '''
    def set_decoder_state(self):
        c_state = F.concat((self[self.lstm_enc[-1]].c, self[self.lstm_rev_enc[-1]].c))
        h_state = F.concat((self[self.lstm_enc[-1]].h, self[self.lstm_rev_enc[-1]].h))
        self[self.lstm_dec[0]].set_state(c_state, h_state)
    '''___QUESTION-1-DESCRIBE-C-END___'''

    '''
    Function to feed an input word through the embedding and lstm layers
        args:
        embed_layer: embeddings layer to use
        lstm_layer:  list of names of lstm layers to use
    '''
    def feed_lstm(self, word, embed_layer, lstm_layer_list, train):
        # get embedding for word
        embed_id = N.dropout(embed_layer(word), ratio=DROPOUT_RATIO, train=train)
        # feed into first LSTM layer
        hs = N.dropout(self[lstm_layer_list[0]](embed_id), ratio=DROPOUT_RATIO, train=train)
        # feed into remaining LSTM layers
        for lstm_layer in lstm_layer_list[1:]:
            hs = N.dropout(self[lstm_layer](hs), ratio=DROPOUT_RATIO, train=train)

    # Function to encode an source sentence word
    def encode(self, word, lstm_layer_list, train):
        self.feed_lstm(word, self.embed_enc, lstm_layer_list, train)

    # Function to decode a target sentence word
    def decode(self, word, train):
        self.feed_lstm(word, self.embed_dec, self.lstm_dec, train)

    def encode_list(self, in_word_list, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # convert list of tokens into chainer variable list
        var_en = (Variable(xp.asarray(in_word_list, dtype=np.int32).reshape((-1,1)),
                           volatile=(not train)))

        var_rev_en = (Variable(xp.asarray(in_word_list[::-1], dtype=np.int32).reshape((-1,1)),
                           volatile=(not train)))

        # array to store hidden states for each word
        enc_states = xp.empty((0,2*self.n_units), dtype=xp.float32)
        first_entry = True

        # encode tokens
        for f_word, r_word in zip(var_en, var_rev_en):
            '''
            ___QUESTION-1-DESCRIBE-D-START___
            Explain why we are performing two encode operations.

            Because we are stepping through the sentence forwards and backwards
            at the same time -- see the call to zip() above -- and we are encoding
            the "forwards" word using the forwards encoder, and the backwards word
            using the backwards encoder.
            '''
            self.encode(f_word, self.lstm_enc, train)
            self.encode(r_word, self.lstm_rev_enc, train)

            '''___QUESTION-1-DESCRIBE-D-END___'''


            # __QUESTION -- Following code is to assist with ATTENTION
            # enc_states stores the hidden state vectors of the encoder
            # this can be used for implementing attention
            if first_entry == False:
                forward_states = F.concat((forward_states, self[self.lstm_enc[-1]].h), axis=0)
                backward_states = F.concat((self[self.lstm_rev_enc[-1]].h, backward_states), axis=0)
            else:
                forward_states = self[self.lstm_enc[-1]].h
                backward_states = self[self.lstm_rev_enc[-1]].h
                first_entry = False

        return enc_states


    # Select a word from a probability distribution
    # should return a chainer variable
    def select_word(self, prob, train=True, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np
        if not sample:
            indx = xp.argmax(prob.data[0])
            pred_word = Variable(xp.asarray([indx], dtype=np.int32), volatile=not train)
        else:
            ''' ___QUESTION-2-SAMPLE '''
            indx = xp.random.choice(a=len(prob.data[0]), p=prob.data[0])
            pred_word = Variable(xp.asarray([indx], dtype=np.int32), volatile=not train)
        return pred_word

    def score(self, enc_state):
        xp = cuda.cupy if self.gpuid >= 0 else np
        return xp.dot(enc_state, xp.transpose(self[self.lstm_dec[-1]].h.data))

    def align(self, enc_states):
        xp = cuda.cupy if self.gpuid >= 0 else np
        alpha = xp.array([xp.exp(self.score(enc_state)) for enc_state in enc_states], dtype=xp.float32)
        alpha /= xp.sum(alpha)
        return alpha

    def encode_decode_train(self, in_word_list, out_word_list, train=True, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        # Add GO_ID, EOS_ID to decoder input
        decoder_word_list = [GO_ID] + out_word_list + [EOS_ID]
        # encode list of words/tokens
        enc_states = self.encode_list(in_word_list, train=train)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode and compute loss
        # convert list of tokens into chainer variable list
        var_dec = (Variable(xp.asarray(decoder_word_list, dtype=np.int32).reshape((-1,1)),
                            volatile=not train))
        # Initialise first decoded word to GOID
        pred_word = Variable(xp.asarray([GO_ID], dtype=np.int32), volatile=not train)

        # compute loss
        self.loss = 0
        # decode tokens
        for next_word_var in var_dec[1:]:
            self.decode(pred_word, train=train)
            if self.attn == NO_ATTN:
                predicted_out = self.out(self[self.lstm_dec[-1]].h)
            else:
                ''' __QUESTION Add attention '''
                alpha = self.align(enc_states.data)
                ctxt = xp.sum(alpha * enc_states.data, axis=0, keepdims=True)
                predicted_out = self.out(self.attn_out(A.concat((ctxt, self[self.lstm_dec[-1]].h))))

            # compute loss
            prob = F.softmax(predicted_out)

            pred_word = self.select_word(prob, train=train, sample=False)
            # pred_word = Variable(xp.asarray([pred_word.data], dtype=np.int32), volatile=not train)
            '''
            ___QUESTION-1-DESCRIBE-E-START___
            Explain what loss is computed with an example. What does this value mean?

            The cross-entropy is a soft measure of how close the network got to the
            correct answer. Here it is used to find how close the predicted word
            (predicted_out) was to the expected word (next_word_var).
            '''
            self.loss += F.softmax_cross_entropy(predicted_out, next_word_var)
            '''___QUESTION-1-DESCRIBE-E-END___'''

        report({"loss":self.loss},self)

        return self.loss

    def decoder_predict(self, start_word, enc_states, max_predict_len=MAX_PREDICT_LEN, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np

        # __QUESTION -- Following code is to assist with ATTENTION
        # alpha_arr should store the alphas for every predicted word
        alpha_arr = xp.empty((0,enc_states.shape[0]), dtype=xp.float32)

        # return list of predicted words
        predicted_sent = []
        # load start symbol
        pred_word = Variable(xp.asarray([start_word], dtype=np.int32), volatile=True)
        pred_count = 0

        # start prediction loop
        while pred_count < max_predict_len and (int(pred_word.data) != (EOS_ID)):
            self.decode(pred_word, train=False)

            if self.attn == NO_ATTN:
                predicted_out = self.out(self[self.lstm_dec[-1]].h)
            else:
                ''' __QUESTION Add attention '''
                alpha = self.align(enc_states.data)
                alpha_arr = xp.concatenate((alpha_arr, xp.transpose(alpha)))
                ctxt = xp.sum(alpha * enc_states.data, axis=0, keepdims=True)
                predicted_out = self.out(self.attn_out(A.concat((ctxt, self[self.lstm_dec[-1]].h))))

            prob = F.softmax(predicted_out)

            pred_word = self.select_word(prob, train=False, sample=sample)
            # add integer id of predicted word to output list
            predicted_sent.append(int(pred_word.data))
            pred_count += 1
        # __QUESTION Add attention
        # When implementing attention, make sure to use alpha_arr to store
        # your attention vectors.
        # The visualisation function in nmt_translate.py assumes such an array as input.
        return predicted_sent, alpha_arr

    def encode_decode_predict(self, in_word_list, max_predict_len=20, sample=False):
        self.reset_state()
        # encode list of words/tokens
        enc_states = self.encode_list(in_word_list, train=False)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode starting with GO_ID
        predicted_sent, alpha_arr = self.decoder_predict(GO_ID, enc_states,
                                                         max_predict_len, sample=sample)
        return predicted_sent, alpha_arr

# In[]:
