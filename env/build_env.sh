conda env create -f environment.yml
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sys2bench
pip install flash_attn==2.7.4.post1 --no-build-isolation
pip install hydra-core==1.3.2
# antlr4-python3-runtime warning is expected