# MIT AI4Code IAP Repair Tutorial

This tutorial is meant to be run on Google's collab. If you'd like to run locally, you should
be on a *nix system (and this was tested on Ubuntu 20.04 via WSL.)

You should navigate to [https://colab.research.google.com/](https://colab.research.google.com/)
and pick to load the `*.ipynb` from github by pointing to the following URL [https://github.com/josepablocam/ai4code_repair](https://github.com/josepablocam/ai4code_repair).


# Local Setup
Download data folder from https://drive.google.com/drive/folders/1U25kzt8I2-pnDVmzPNs8s4o7h_ePfjL1?usp=sharing (or do so by executing
notebook).

Create a virtual environment and setup installation.

```bash
python -m virtualenv env/
source env/bin/activate
source install.sh
```

# Cites
* We use the DeepFix dataset, courtesy of Gupta et al (Gupta, Rahul, et al. "Deepfix: Fixing common c language errors by deep learning." Proceedings of the aaai conference on artificial intelligence. Vol. 31. No. 1. 2017.), available for download at [https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip](https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip)
* Some of our utility code is also implemented based on code snippets in the [DeepFix repository](https://bitbucket.org/iiscseal/deepfix/src/master/)

# TODOs:
* Finish [TODO] and [CITE] places in tutorial text
* Fix collab setup
* Run final fine-tuning run and store model in folder as well

https://drive.google.com/file/d/1V5sRePMY6D3IEaj3mMaIOYsp_JA6Apy6/view?usp=share_link