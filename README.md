# MIT AI4Code IAP Repair Tutorial

TODO: write


# Setup
Run in *nix (test in WSL)

Download data folder from  https://drive.google.com/drive/folders/1lJUFDmIcXu_ZcOUKBdY2iSvVT4_PIKZc?usp=sharing

Store in `./data/`


```bash
python -m virtualenv env/
source env/bin/activate
bash install.sh
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


for C99 grammar
https://gist.github.com/codebrainz/2933703

# TODOs:
* Fix collab setup
* Run final fine-tuning run and store model in folder as well
* Adapt GRMTOOLs .l and .y 
* Add text
* Clean up tutorial and add references/pointers for more problems
* Add tests
* Any other "stretch" task

# FIXME: are locations participants can try some extensions :)
