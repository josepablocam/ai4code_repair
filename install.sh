
# setting up Rust for grmtools
curl https://sh.rustup.rs -sSf | bash -s -- -y
source "$HOME/.cargo/env"

# install grmtools
pushd resources/
git clone git@github.com:softdevteam/grmtools.git
pushd grmtools
cargo build --release
export PATH=$PATH:$(readlink -f target/release)
popd

# install our package
pip install -r requirements.txt
pip install -e .