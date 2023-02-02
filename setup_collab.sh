if [ ! -d repair ]
then
    echo "Running from colab"
    git clone https://github.com/josepablocam/ai4code_repair.git
    cd ai4code_repair
    source install.sh
else
    echo "Running from local setup -- please make sure you setup using README"
fi