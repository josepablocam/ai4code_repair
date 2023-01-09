# MIT AI4Code IAP Repair Tutorial

TODO: write


# Setup
Run in *nix (test in WSL)

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
* Setup full evaluation for benchmarks
* Setup few-shot section for Codex
    - Fixed few-shots
    - Random few-shots
    - CodeBERT-based few-shots

* Add CodeT5 portion
* Add text for current portion of tutorial
