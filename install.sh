
# setting up Rust for grmtools
echo "Setting up Rust"
curl https://sh.rustup.rs -sSf | bash -s -- -y
source "$HOME/.cargo/env"

# install grmtools
echo "Setting up grmtools"
pushd resources/
git clone https://github.com/softdevteam/grmtools.git
pushd grmtools
cargo build --release
export PATH=$PATH:$(readlink -f target/release)
popd
popd

# install sqlite3
apt install sqlite3

# install our package
echo "Setting up ai4code_repair"
pip install -r requirements.txt
pip install -e .