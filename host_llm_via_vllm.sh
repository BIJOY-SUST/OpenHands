# 1. Create and activate a new conda environment
conda create -n vllm_env_v3 python=python=3.12
echo "vllm environment created successfully"
conda activate vllm_env_v3
echo "vllm environment activated successfully"

# 2. Install vllm using pip
pip install vllm
pip install --upgrade poetry
pip install --upgrade jinja2
pip install --upgrade openai

conda deactivate
echo "vllm environment deactivated successfully"
conda activate vllm_env_v3
# pip list
echo "vllm installed successfully"
# 3. Start the vllm server with Llama 3.1 8B Instruct
CUDA_VISIBLE_DEVICES=2 vllm serve \
  /shared_archive/common/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ \
  --dtype auto \
  --api-key token-abc123 \
  --gpu-memory-utilization 0.9 \
  --port 8002