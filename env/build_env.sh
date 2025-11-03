conda env create -f env/environment.yml
source $(conda info --base)/etc/profile.d/conda.sh
conda activate reasoning_env
pip install flash_attn==2.7.4.post1 --no-build-isolation