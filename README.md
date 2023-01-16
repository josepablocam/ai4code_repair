# MIT AI4Code IAP Repair Tutorial

TODO: write


# Setup
Run in *nix (test in WSL)

Download data folder from  https://drive.google.com/drive/folders/1lJUFDmIcXu_ZcOUKBdY2iSvVT4_PIKZc?usp=sharing

Store in `./data/`


```bash
python -m virtualenv env/
source env/bin/activate
pip install -r requirements.txt
pip install -e .
```


# Sources

* For C syntax repair data
wget https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip -o deepfix.zip
unzip deepfix.zip
mv prutor*/*.gz .
gunzip *.gz

https://bitbucket.org/iiscseal/deepfix/src/master/
@misc{deepfix2017repository,
author = {Gupta, Rahul and Pal, Soham and Kanade, Aditya and Shevade, Shirish},
title = "DeepFix: Fixing Common C Language Errors by Deep Learning",
year = "2017",
url = "http://www.iisc-seal.net/deepfix",
note = "[Online; accessed 01-08-2023]"
}


# TODOs:
* Finetuned CodeT5 section
* Add text
* Add grmtools section: (need to add C .y file and then can play with it, extra for students would be
to turn parser recovery suggestion into repair, and run in benchmark)
* Clean up tutorial and add references/pointers for more problems
* Add tests
* Setup skeleton for fine-tuning CodeBERT (to perform target similarity tuning)
* Any other "stretch" task
* Pin versions of libs
* Try out in colab

# FIXME: are locations participants can try some extensions :)
