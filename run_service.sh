export WEBSOCKET_ADDRESS=ws://172.17.0.1:8765
export AI_NO=AI_000191011
export SERVICE_GPU_CONFIG=0:4
uvicorn service:app --ws-ping-interval 300 --ws-ping-timeout 300 --host 0.0.0.0 --port 8000
