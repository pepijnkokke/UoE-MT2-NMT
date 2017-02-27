latest code available in [github](https://github.com/0xSameer/chainer-code/tree/nmt)

Start with the file:
**nmt_setup.ipynb** --- this gives a quick peak at the dataset and the input files provided.

Remaining files:
- enc_dec.ipynb       --- implementation of encoder-decoder with global attention
- nmt_translate.ipynb --- source code to load data and run training, compute bleu
- **nmt_config.py**       --- neural model parameters, data selection, etc.

For running multiple experiments, we are also providing the python versions of these IPython notebooks.

**Experiment Procedure**
- set experiment parameters in *nmt_config.py*
- execute bash script **run_exp.bat**



