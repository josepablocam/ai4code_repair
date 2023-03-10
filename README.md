# MIT AI4Code IAP Repair Tutorial

This tutorial is meant to be run on Google colab. If you'd like to run locally, you should
be on a *nix system (and this was tested on Ubuntu 20.04 via WSL.)

The original version of this tutorial was given as part of the "AI for Code and Science" IAP (independent activities period) at MIT in Spring 2023.

You should navigate to [https://colab.research.google.com/](https://colab.research.google.com/)
and pick to load the `*.ipynb` from github by pointing to the following URL [https://github.com/josepablocam/ai4code_repair](https://github.com/josepablocam/ai4code_repair).

Once you do, you will want to set your runtime to have access to a GPU (otherwise some portions
will be quite slow -- but doable, as long as you don't do the finetuning portion of 
the tutorial).

Go to `Runtime` > `Change runtime type` > `GPU` and then click `Connect`.

Note that some of the setup is run using `%%bash` and colab seems to accumulate stdout
messages until it is done, so you may want to be a bit patient.

# Known Issues
* Inference isn't batched so any use of neural models is pretty slow -- I will try to batch at some point if I get a chance to put more time into this (otherwise, PRs welcomed).

# Cites
* We use the DeepFix dataset, courtesy of Gupta et al (Gupta, Rahul, et al. "Deepfix: Fixing common c language errors by deep learning." Proceedings of the aaai conference on artificial intelligence. Vol. 31. No. 1. 2017.), available for download at [https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip](https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip)
* Some of our utility code is also implemented based on code snippets in the [DeepFix repository](https://bitbucket.org/iiscseal/deepfix/src/master/)

Below cites referenced in the tutorial itself.

[1] Gupta, Rahul, et al. "Deepfix: Fixing common c language errors by deep learning." Proceedings of the aaai conference on artificial intelligence. Vol. 31. No. 1. 2017.

[2] Wang, Yue, et al. "Codet5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation." arXiv preprint arXiv:2109.00859 (2021).

[3] Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." The Journal of Machine Learning Research 21.1 (2020): 5485-5551.

[4] Diekmann, Lukas, and Laurence Tratt. "Don't Panic! Better, Fewer, Syntax Errors for LR Parsers." arXiv preprint arXiv:1804.07133 (2018).

[5] Chen, Mark, et al. "Evaluating large language models trained on code." arXiv preprint arXiv:2107.03374 (2021).

[6] Feng, Zhangyin, et al. "Codebert: A pre-trained model for programming and natural languages." arXiv preprint arXiv:2002.08155 (2020).

[7] Johnson, Jeff, Matthijs Douze, and Herv?? J??gou. "Billion-scale similarity search with gpus." IEEE Transactions on Big Data 7.3 (2019): 535-547.

[8] Yasunaga, Michihiro, and Percy Liang. "Break-it-fix-it: Unsupervised learning for program repair." International Conference on Machine Learning. PMLR, 2021.

