# coding: utf-8
#---------------------------------------------------------------------
'''
Neural Machine Translation - Translation experiments
    Training, evaluation and prediction functions
'''
#---------------------------------------------------------------------

# In[ ]:

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from tqdm import tqdm
import sys
import os
from collections import Counter
import math
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import time
import matplotlib.gridspec as gridspec
import importlib
# %matplotlib inline

# In[ ]:

#---------------------------------------------------------------------
# Load configuration
#---------------------------------------------------------------------
from nmt_config import *

# In[ ]:
#---------------------------------------------------------------------
# Load encoder decoder model definition
#---------------------------------------------------------------------
from enc_dec import *

# In[ ]:

xp = cuda.cupy if gpuid >= 0 else np

# In[ ]:
#---------------------------------------------------------------------
# Load dataset
#---------------------------------------------------------------------
w2i = pickle.load(open(w2i_path, "rb"))
i2w = pickle.load(open(i2w_path, "rb"))
vocab = pickle.load(open(vocab_path, "rb"))
vocab_size_en = min(len(i2w["en"]), max_vocab_size["en"])
vocab_size_fr = min(len(i2w["fr"]), max_vocab_size["fr"])
print("vocab size, en={0:d}, fr={1:d}".format(vocab_size_en, vocab_size_fr))
print("{0:s}".format("-"*50))
#---------------------------------------------------------------------

# In[ ]:
#---------------------------------------------------------------------
# Set up model
#---------------------------------------------------------------------
model = EncoderDecoder(vocab_size_fr, vocab_size_en,
                       num_layers_enc, num_layers_dec,
                       hidden_units, gpuid, attn=use_attn)
if gpuid >= 0:
    cuda.get_device(gpuid).use()
    model.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(model)

'''
___QUESTION-1-DESCRIBE-F-START___

- Describe what the following line of code does
'''
optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=5))
'''___QUESTION-1-DESCRIBE-F-END___'''

# In[ ]:

#---------------------------------------------------------------------
print("Training progress will be logged in:\n\t{0:s}".format(log_train_fil_name))
print("{0:s}".format("-"*50))
print("Trained model will be saved as:\n\t{0:s}".format(model_fil))
print("{0:s}".format("-"*50))
#---------------------------------------------------------------------

# In[ ]:
#---------------------------------------------------------------------
# Evaluation utilities
#---------------------------------------------------------------------
# Compute perplexity over validation set
def compute_dev_pplx():
    loss = 0
    num_words = 0
    num_sents = 0
    with open(text_fname["fr"], "rb") as fr_file, open(text_fname["en"], "rb") as en_file:
        with tqdm(total=NUM_DEV_SENTENCES) as pbar:
            sys.stderr.flush()
            out_str = "loss={0:.6f}".format(0)
            pbar.set_description(out_str)
            for i, (line_fr, line_en) in enumerate(zip(fr_file, en_file), start=1):
                if i > NUM_TRAINING_SENTENCES and i <= (NUM_TRAINING_SENTENCES + NUM_DEV_SENTENCES):
                    fr_sent = line_fr.strip().split()
                    en_sent = line_en.strip().split()

                    fr_ids = [w2i["fr"].get(w, UNK_ID) for w in fr_sent]
                    en_ids = [w2i["en"].get(w, UNK_ID) for w in en_sent]

                    # compute loss
                    curr_loss = float(model.encode_decode_train(fr_ids, en_ids, train=False).data)  / len(en_ids)
                    loss += curr_loss
                    num_words += len(en_ids)

                    out_str = "loss={0:.6f}".format(curr_loss)
                    pbar.set_description(out_str)
                    pbar.update(1)
                    num_sents += 1
                
            # end for
        # end pbar
    # end with open file
    #loss_per_word = loss / num_words
    loss_per_sentence = loss / num_sents
    pplx = 2 ** loss_per_sentence

    print("{0:s}".format("-"*50))
    print("{0:14s} | {1:.4f}".format("dev perplexity", pplx))
    print("{0:14s} | {1:d}".format("# words in dev", num_words))
    print("{0:s}".format("-"*50))

    return pplx

# In[ ]:
#---------------------------------------------------------------------
# Compute Bleu score
#---------------------------------------------------------------------
def bleu_stats(hypothesis, reference):
    yield len(hypothesis)
    yield len(reference)
    for n in range(1,5):
        s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)+1-n)])
        r_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)+1-n)])
        yield max([sum((s_ngrams & r_ngrams).values()), 0])
        yield max([len(hypothesis)+1-n, 0])

# Compute BLEU from collected statistics obtained by call(s) to bleu_stats
def bleu(stats):
    if len(list(filter(lambda x: x==0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum([math.log(float(x)/y) for x,y in zip(stats[2::2],stats[3::2])]) / 4.
    return math.exp(min([0, 1-float(r)/c]) + log_bleu_prec)

def compute_dev_bleu():
    list_of_references = []
    list_of_hypotheses = []
    with open(text_fname["fr"], "rb") as fr_file, open(text_fname["en"], "rb") as en_file:
        for i, (line_fr, line_en) in enumerate(zip(fr_file, en_file), start=1):
            if i > NUM_TRAINING_SENTENCES and i <= (NUM_TRAINING_SENTENCES + NUM_DEV_SENTENCES):
                fr_sent = line_fr.strip().split()
                en_sent = line_en.strip().split()

                fr_ids = [w2i["fr"].get(w, UNK_ID) for w in fr_sent]
                en_ids = [w2i["en"].get(w, UNK_ID) for w in en_sent]

                list_of_references.append(line_en.strip().decode())
                pred_sent, alpha_arr = model.encode_decode_predict(fr_ids, max_predict_len=MAX_PREDICT_LEN)
                pred_words = [i2w["en"][w].decode() for w in pred_sent if w != EOS_ID]
                pred_sent_line = " ".join(pred_words)
                list_of_hypotheses.append(pred_sent_line)
            if i > (NUM_TRAINING_SENTENCES + NUM_DEV_SENTENCES):
                break

    stats = [0 for i in range(10)]
    for (r,h) in zip(list_of_references, list_of_hypotheses):
        stats = [sum(scores) for scores in zip(stats, bleu_stats(h,r))]

    bleu_score = (100 * bleu(stats))
    print("BLEU: {0:0.3f}".format(bleu_score))
    return bleu_score

# In[ ]:
#---------------------------------------------------------------------
# Main training loop
#---------------------------------------------------------------------
def train_loop(text_fname, num_training, num_epochs, log_mode="a"):
    # Set up log file for loss
    log_train_fil = open(log_train_fil_name, mode=log_mode)
    log_train_csv = csv.writer(log_train_fil, lineterminator="\n")

    sys.stderr.flush()

    for epoch in range(num_epochs):
        with open(text_fname["fr"], "rb") as fr_file, open(text_fname["en"], "rb") as en_file:
            with tqdm(total=num_training) as pbar:
                sys.stderr.flush()
                loss_per_epoch = 0
                out_str = "epoch={0:d}, iter={1:d}, loss={2:.6f}, mean loss={3:.6f}".format(
                                epoch+1, 0, 0, 0)
                pbar.set_description(out_str)

                for i, (line_fr, line_en) in enumerate(zip(fr_file, en_file), start=1):
                    fr_sent = line_fr.strip().split()
                    en_sent = line_en.strip().split()

                    fr_ids = [w2i["fr"].get(w, UNK_ID) for w in fr_sent]
                    en_ids = [w2i["en"].get(w, UNK_ID) for w in en_sent]

                    it = (epoch * NUM_TRAINING_SENTENCES) + i

                    if i > num_training:
                        break

                    # compute loss
                    loss = model.encode_decode_train(fr_ids, en_ids)

                    # clear gradient
                    model.cleargrads()
                    # backprop
                    loss.backward()
                    # update parameters
                    optimizer.update()
                    # store loss value for display
                    loss_val = float(loss.data) / len(en_ids)
                    loss_per_epoch += loss_val

                    out_str = "epoch={0:d}, iter={1:d}, loss={2:.6f}, mean loss={3:.6f}".format(
                               epoch+1, it, loss_val, (loss_per_epoch / i))
                    pbar.set_description(out_str)
                    pbar.update(1)

                    # log every 100 sentences
                    if i % 100 == 0:
                        log_train_csv.writerow([it, loss_val])


        # compute precision, recall and F-score
        metrics = predict(s=NUM_TRAINING_SENTENCES, 
                         num=NUM_DEV_SENTENCES, display=False, plot=False)
        prec = np.sum(metrics["cp"]) / np.sum(metrics["tp"])
        rec = np.sum(metrics["cp"]) / np.sum(metrics["t"])
        f_score = 2 * (prec * rec) / (prec + rec)

        print("{0:s}".format("-"*50))
        print("{0:10s} | {1:0.4f}".format("precision", prec))
        print("{0:10s} | {1:0.4f}".format("recall", rec))
        print("{0:10s} | {1:0.4f}".format("f1", f_score))
        print("{0:s}".format("-"*50))
        print("computing perplexity")
        pplx = compute_dev_pplx()

        # Backup model every epoch
        print("Saving model")
        serializers.save_npz(model_fil, model)
        print("Finished saving model")
        print("{0:s}".format("-"*50))

        # Compute Bleu every 2 epochs
        if epoch % 2 == 0:
            print("computing bleu")
            bleu_score = compute_dev_bleu()
            print("finished computing bleu ... ")
            print("{0:s}".format("-"*50))
        
    # At the end of training, make some predictions
    # make predictions over both training and dev sets
    print("Training set predictions")
    _ = predict(s=0, num=3, plot=False)
    print("{0:s}".format("-"*50))
    print("dev set predictions")
    _ = predict(s=NUM_TRAINING_SENTENCES, num=3, plot=False)
    print("{0:s}".format("-"*50))

    print("{0:s}".format("-"*50))
    # Check if Bleu needs to be recomputed
    if num_epochs % 2 != 0:
        bleu_score = compute_dev_bleu()
        print("{0:s}".format("-"*50))
    print("{0:s}".format("-"*50))

    # close log file
    log_train_fil.close()
    print("Finished training. Filenames:")
    print(log_train_fil_name)
    print(model_fil)

# In[ ]:
#---------------------------------------------------------------------
# __QUESTION -- Following code is to assist with ATTENTION
#---------------------------------------------------------------------
from matplotlib.font_manager import FontProperties
'''
Support function to plot attention vectors
'''
def plot_attention(alpha_arr, fr, en, plot_name=None):
    if gpuid >= 0:
        alpha_arr = cuda.to_cpu(alpha_arr).astype(np.float32)

    #alpha_arr /= np.max(np.abs(alpha_arr),axis=0)
    fig = plt.figure()
    fig.set_size_inches(8, 8)

    gs = gridspec.GridSpec(2, 2, width_ratios=[12,1],height_ratios=[12,1])

    ax = plt.subplot(gs[0])
    ax_c = plt.subplot(gs[1])

    cmap = sns.light_palette((200, 75, 60), input="husl", as_cmap=True)
    prop = FontProperties(fname='fonts/IPAfont00303/ipam.ttf', size=12)
    ax = sns.heatmap(alpha_arr, xticklabels=fr, yticklabels=en, ax=ax, cmap=cmap, cbar_ax=ax_c)

    ax.xaxis.tick_top()
    ax.yaxis.tick_right()

    ax.set_xticklabels(en, minor=True, rotation=60, size=12)
    for label in ax.get_xticklabels(minor=False):
        label.set_fontsize(12)
        label.set_font_properties(prop)

    for label in ax.get_yticklabels(minor=False):
        label.set_fontsize(12)
        label.set_rotation(-90)
        label.set_horizontalalignment('left')

    ax.set_xlabel("Source", size=20)
    ax.set_ylabel("Hypothesis", size=20)

    if plot_name:
        fig.savefig(plot_name, format="png")

# In[ ]:

#---------------------------------------------------------------------
# Helper function for prediction
#---------------------------------------------------------------------
def predict_sentence(line_fr, line_en=None, display=True, 
                     plot_name=None, p_filt=0, r_filt=0, sample=False):
    fr_sent = line_fr.strip().split()
    fr_ids = [w2i["fr"].get(w, UNK_ID) for w in fr_sent]
    # english reference is optional. If provided, compute precision/recall
    if line_en:
        en_sent = line_en.strip().split()
        en_ids = [w2i["en"].get(w, UNK_ID) for w in en_sent]

    pred_ids, alpha_arr = model.encode_decode_predict(fr_ids, 
                                                      max_predict_len=MAX_PREDICT_LEN,
                                                      sample=sample)
    pred_words = [i2w["en"][w].decode() for w in pred_ids]

    prec = 0
    rec = 0
    filter_match = False

    matches = count_match(en_ids, pred_ids)
    prec = matches/len(pred_ids)
    rec = matches/len(en_ids)

    if display and (prec >= p_filt and rec >= r_filt):
        filter_match = True
        # convert raw binary into string
        fr_words = [w.decode() for w in fr_sent]

        print("{0:s}".format("-"*50))
        print("{0:s} | {1:80s}".format("Src", line_fr.strip().decode()))
        print("{0:s} | {1:80s}".format("Ref", line_en.strip().decode()))
        print("{0:s} | {1:80s}".format("Hyp", " ".join(pred_words)))

        print("{0:s}".format("-"*50))

        print("{0:s} | {1:0.4f}".format("precision", prec))
        print("{0:s} | {1:0.4f}".format("recall", rec))

        if plot_name and use_attn:
            plot_attention(alpha_arr, fr_words, pred_words, plot_name)

    return matches, len(pred_ids), len(en_ids), filter_match

# In[ ]:
#---------------------------------------------------------------------
'''
Function to make predictions.
s       : starting index of the line in the parallel data from which to make predictions
num     : number of lines starting from "s" to make predictions for
plot    : plot attention if True
display : whether to display additional info
p_filt  : precision filter. Only displays predicted sentences with precision above p_filt
r_filt  : recall filter. Only displays predicted sentences with recall above p_filt
sample  : if False, predict word with maximum probability in the model, else sample
'''
#---------------------------------------------------------------------
def predict(s=NUM_TRAINING_SENTENCES, num=NUM_DEV_SENTENCES, 
            display=True, plot=False, p_filt=0, r_filt=0, sample=False):
    
    if display:
        print("English predictions, s={0:d}, num={1:d}:".format(s, num))

    metrics = {"cp":[], "tp":[], "t":[]}

    filter_count = 0

    with open(text_fname["fr"], "rb") as fr_file, open(text_fname["en"], "rb") as en_file:
        for i, (line_fr, line_en) in enumerate(zip(fr_file, en_file), start=0):
            if i >= s and i < (s+num):
                if plot:
                    plot_name = os.path.join(model_dir, "sample_{0:d}_plot.png".format(i+1))
                else:
                    plot_name=None

                # make prediction
                cp, tp, t, f = predict_sentence(line_fr,
                                             line_en,
                                             display=display,
                                             plot_name=plot_name, 
                                             p_filt=p_filt, r_filt=r_filt,
                                             sample=sample)
                metrics["cp"].append(cp)
                metrics["tp"].append(tp)
                metrics["t"].append(t)
                filter_count += (1 if f else 0)

    if display:
        print("sentences matching filter = {0:d}".format(filter_count))
    
    return metrics


# In[ ]:
#---------------------------------------------------------------------
# Helper function to compute precision recall
#---------------------------------------------------------------------
def count_match(list1, list2):
    # each list can have repeated elements. The count should account for this.
    count1 = Counter(list1)
    count2 = Counter(list2)
    count2_keys = count2.keys()-set([UNK_ID])
    common_w = set(count1.keys()) & set(count2_keys)
    matches = sum([min(count1[w], count2[w]) for w in common_w])
    return matches

# In[ ]:

#---------------------------------------------------------------------
# Main -- program execution starts here
#---------------------------------------------------------------------

def main():
    print("Existing model {0:s}".format("found" if os.path.exists(model_fil) else "not found!"))
    print("{0:s}".format("-"*50))
    if os.path.exists(model_fil):
        if load_existing_model:
            print("loading model ...")
            serializers.load_npz(model_fil, model)
            print("finished loading: {0:s}".format(model_fil))
            print("{0:s}".format("-"*50))
        else:
            print("""model file already exists!!
                Delete before continuing, or enable load_existing flag""".format(model_fil))
            return

    if NUM_EPOCHS > 0:
        train_loop(text_fname, NUM_TRAINING_SENTENCES, NUM_EPOCHS)
    
if __name__ == "__main__":
    main()

#---------------------------------------------------------------------

def test_lam_tran():
    lines_fr = ["君 は １ 日 で それ が でき ま す か 。", "皮肉 な 笑い を 浮かべ て 彼 は 私 を 見つめ た 。", "あなた は 午後 何 を し た い で す か 。"]
    lines_en = ["can you do it in one day ?", "he stared at me with a satirical smile .", "what do you want to do in the afternoon ?" ]

    for line_fr, line_en in zip(lines_fr, lines_en):
        line_fr = line_fr.encode()
        line_en = line_en.encode()

        predict_sentence(line_fr=line_fr, line_en=line_en)


