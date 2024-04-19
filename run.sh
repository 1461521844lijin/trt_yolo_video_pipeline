./build/test_yolo_detect rtmp://video-pull.eflyyzh.com/UAS057128001/UAS057128001_2024_04_18_09_09_15 rtmp://video-push.eflyyzh.com/UAS057128001/UAS057128001_2024_04_18_09_09_15-AISTREAM http://127.0.0.1:8899/item/ task_no:10000
# export WEBSOCKET_ADDRESS=ws://172.17.0.1:8765/
# export AI_NO=10000
# export SERVICE_GPU_CONFIG=0:1
# uvicorn service:app --ws-ping-interval 300 --ws-ping-timeout 300 --host 0.0.0.0 --port 8000
# uvicorn backend_app:app --port 8899 --host 0.0.0.0
#rtmp://video-pull.eflyyzh.com/0403/1
#rtmp://video-push.eflyyzh.com/ai/test
#rtmp://video-pull.eflyyzh.com/src/test
#rtmp://video-pull.eflyyzh.com/ai/test