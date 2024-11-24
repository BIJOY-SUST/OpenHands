CUDA_VISIBLE_DEVICES=0 vllm serve \
  /work/10363/bbijoy2024/ls6/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/ \
  --dtype auto \
  --api-key token-abc123 \
  --gpu-memory-utilization 0.9 \
  --port 8002