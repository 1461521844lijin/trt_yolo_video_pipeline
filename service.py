import time

from fastapi import FastAPI
from websockets.legacy.client import WebSocketClientProtocol
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Optional
import asyncio
import json
import websockets
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler

GPU_CONFIG_PATH = './config.yaml'

if "WEBSOCKET_ADDRESS" not in os.environ:
    raise RuntimeError("WEBSOCKET_ADDRESS must be set")
WEBSOCKET_ADDRESS = os.environ["WEBSOCKET_ADDRESS"]

if "AI_NO" not in os.environ:
    raise RuntimeError("AI_NO must be set")
AI_NO = os.environ["AI_NO"]
support_image = ['bmp', 'jpg', 'JPG', 'jpeg', 'bmp', 'png']

def __get_global_consumer_states(config_path=GPU_CONFIG_PATH):
    import os
    if 'SERVICE_GPU_CONFIG' in os.environ:
        data = os.environ['SERVICE_GPU_CONFIG']
        data = data.strip().replace(' ', '')
        data = data.split(',')
        data = [kv.split(':') for kv in data]
        data = {int(k): int(v) for k, v in data}
    else:
        import yaml
        data = yaml.load(open(config_path).read(), yaml.Loader)
    global_state = {(gpu_id, id): None for gpu_id, counts in data.items() for id in range(counts)}

    return global_state


global_consumer_states_lock = asyncio.Lock()
global_consumer_states = __get_global_consumer_states()
global_camera_pipeline = {}

global_sch_srv = None
global_tasks_queue = asyncio.Queue()


async def run_command(args, task_no, gpu_id, id):
    print(f"####################\n{args}")
    #process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, env={"CUDA_VISIBLE_DEVICES": str(gpu_id)})
    process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, env={"CUDA_VISIBLE_DEVICES": str(gpu_id)})
    print(f"pid is: {process.pid}");
    async with global_consumer_states_lock:
        global_consumer_states[(gpu_id, id)] = task_no
        global_camera_pipeline[task_no] = (process.pid, gpu_id, id)
    stdout, stderr = await process.communicate()

    await process.wait()


    # if stdout:
    #     print(f'[stdout]\n{stdout.decode()}')
    # if stderr:
    #     print(f'[stderr]\n{stderr.decode()}')
    #print(f"stdout: {stdout.decode().strip()}, stderr: {stderr.decode().strip()}")
    async with global_consumer_states_lock:
        global_consumer_states[(gpu_id, id)] = None
        if task_no in global_camera_pipeline:
            del global_camera_pipeline[task_no]
    return stdout.decode().strip() if stdout is not None else ''


@dataclass
class Task:
    video_pull_url: str
    video_push_url: str
    keyframe_push_url: str
    task_no: str
    ai_type: List[str]
    # 新增 2023-08-18
    video_pull_type: str
    # 新增 2023-08-22
    uav_no: str
    fly_no: str


app = FastAPI()


async def task_executor(gpu_id, id):
    while True:
        task: Task = await global_tasks_queue.get()
        print(f"{task.video_pull_url}")
        print(f"{task.video_push_url}")
        print(f"{task.keyframe_push_url}")
        print(f"{task.task_no}")
        print(f"{task.ai_type}")
        print(f"{task.video_pull_type}")
        if task.video_pull_type != "image":
            cmd =  ["./build/test_yolo_detect", f"{task.video_pull_url}", f"{task.video_push_url}",]
            cmd += ["-p", f"{task.keyframe_push_url}"]
            cmd += [f"{task.task_no}"]
            cmd += ["-m", f"2"]
            #filter_out_classes = []
            # if len(task.ai_type) != 3:
            #     filter_out_classes = []
            #     if "people" not in task.ai_type:
            #         filter_out_classes.append("0")
            #     if "car" not in task.ai_type:
            #         filter_out_classes.append("1")
            #     if "animal" not in task.ai_type:
            #         filter_out_classes.append("2")
            # cmd += ["-f", f"{';'.join(filter_out_classes)}"]
            await run_command(cmd, task.task_no, gpu_id, id)
        else:
            cmd = ["python3", "ai-infer/ai_detect_v1.py", "--file-path", f"{task.video_pull_url}"]
            cmd += ["-o", f"{task.video_push_url}"]
            cmd += ["--push-address", f"{task.keyframe_push_url}"]
            cmd += ["--camera-id", f"{task.task_no}"]
            cmd += ["--uav-no", f"{task.uav_no}"]
            cmd += ["--fly", f"{task.fly_no}"]


            if len(task.ai_type) != 4:
                filter_out_classes = []
                if "v_crack" not in task.ai_type:
                    filter_out_classes.append("0")
                if "h_crack" not in task.ai_type:
                    filter_out_classes.append("1")
                if "d_crack" not in task.ai_type:
                    filter_out_classes.append("2")
                if "m_crack" not in task.ai_type:
                    filter_out_classes.append("3")
                cmd += ["--categories", f"{';'.join(filter_out_classes)}"]
            time.sleep(1)
            imageresult = await run_command(cmd, task.task_no, gpu_id, id)
            with open("logxxx.txt", 'w') as f:
                f.write(imageresult)
            index = imageresult.find('keyframe')
            res = imageresult[index+9:]

            #todo 发送完成命令
            await manager.ai_completed(taskNo=task.task_no, videl_pull_url=task.video_pull_url, keyframe_id=res)



class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocketClientProtocol] = []
        self.connection_lock = asyncio.Lock()

    async def connect(self, websocket: WebSocketClientProtocol):
        async with self.connection_lock:
            self.active_connections.append(websocket)
        resource_msg = await query_resource()
        message = {
            "action": "AI_CONNECT",
            "data": {
                "ai_no": AI_NO,
                "task_processing": resource_msg.running_camera_pipeline
            }
        }
        await self.send_personal_message(message, websocket)
        response = await self.recv_personal_message(websocket)
        if response["status"] != 0:
            print(f"failed connectting: {response}\n{message}")
        else:
            print(f"successfully connectting: {message}")

    async def disconnect(self, websocket: WebSocketClientProtocol, send_disconnect: bool = True):
        if send_disconnect:
            message = {
                "action": "AI_DISCONNECT",
                "data": {
                }
            }
            try:
                await self.send_personal_message(message, websocket)
            except:
                pass
                
        async with self.connection_lock:
            await asyncio.wait_for(websocket.close(), timeout=2)
            try:
                self.active_connections.remove(websocket)
            except ValueError:
                pass

    async def disconnect_all(self):
        async with self.connection_lock:
            for ws in list(self.active_connections):
                await self.disconnect(ws)
                
    async def send_personal_message(self, message: dict, websocket: WebSocketClientProtocol):
        try:
            if websocket.closed:
                raise Exception("ws closed")
            await websocket.send(json.dumps(message))
        except Exception:
            await self.disconnect(websocket, send_disconnect=False)

    async def recv_personal_message(self, websocket: WebSocketClientProtocol):
        try:
            if websocket.closed:
                raise Exception("ws closed")
            response = await websocket.recv()
            return json.loads(response)
        except BaseException:
            await self.disconnect(websocket, send_disconnect=False)
        return {"action": "pong"}

    async def ping_pong(self):
        failed_connection = []
        async with self.connection_lock:
            for ws in list(self.active_connections):
                try:
                    if ws.closed:
                        raise Exception("ws closed")
                    else:
                        await asyncio.wait_for(self.send_personal_message({"action": "ping"}, ws), timeout=2)
                except Exception:
                    failed_connection.append(ws)
            for ws in failed_connection:
                await self.disconnect(ws)

    async def ai_completed(self,taskNo:str,videl_pull_url:str,keyframe_id):
        print("aicomplate")
        failed_connection = []
        data={
            'task_no':taskNo,
            'video_pull_url':videl_pull_url,
            'keyframe_id':keyframe_id
        }
        async with self.connection_lock:
            for ws in list(self.active_connections):
                try:
                    if ws.closed:
                        raise Exception("ws closed")
                    else:
                        await asyncio.wait_for(self.send_personal_message({"action": "AI_COMPLETED","data":data}, ws), timeout=2)
                except Exception:
                    failed_connection.append(ws)
            for ws in failed_connection:
                await self.disconnect(ws)


manager = ConnectionManager()


class CreatePipelineRequest(BaseModel):
    # video_pull_url: str
    # video_push_url: str
    # ai_type: List[str]
    # keyframe_push_url: str
    # task_no: str
    video_pull_url: str
    video_push_url: str
    keyframe_push_url: str
    task_no: str
    ai_type: List[str]
    # 新增 2023-08-18
    video_pull_type: str
    # 新增 2023-08-22
    uav_no: str
    fly_no: str


class CreatePipelineResponse(BaseModel):
    status: int
    video_push_url: str
    task_no: str


class QueryResourceResponse(BaseModel):
    status: int
    busy: int
    free: int
    running_camera_pipeline: List[str]


class ClosePipelineRequest(BaseModel):
    task_no: str


class ClosePipelineResponse(BaseModel):
    task_no: str
    status: int


class WebSocketRequest(BaseModel):
    action: str
    data: dict


class WebSocketResponse(BaseModel):
    action: str
    data: dict
    status: Optional[int]
    message: Optional[str]


async def create_pipeline(item: CreatePipelineRequest):
    async with global_consumer_states_lock:
        if item.task_no in global_camera_pipeline:
            response = CreatePipelineResponse(
                status=400,
                video_push_url='',
                task_no=item.task_no
            )
            return response
        states = list(global_consumer_states.values())
        busy = sum([1 for state in states if state is not None])
        if len(states) - busy == 0:
            response = CreatePipelineResponse(
                status=406,
                video_push_url='',
                task_no=item.task_no
            )
            return response

    await global_tasks_queue.put(
        Task(
            # video_pull_url=item.video_pull_url,
            # video_push_url=item.video_push_url,
            # ai_type=item.ai_type,
            # keyframe_push_url=item.keyframe_push_url,
            # task_no=item.task_no,
            video_pull_url=item.video_pull_url,
            video_pull_type=item.video_pull_type,
            video_push_url=item.video_push_url,
            ai_type=item.ai_type,
            keyframe_push_url=item.keyframe_push_url,
            task_no=item.task_no,
            uav_no=item.uav_no,
            fly_no=item.fly_no
        )
    )
    response = CreatePipelineResponse(
        status=200,
        video_push_url=item.video_push_url,
        task_no=item.task_no
    )
    return response


async def query_resource():
    async with global_consumer_states_lock:
        states = list(global_consumer_states.values())
        running_camera_pipeline = list([state for state in states if state is not None])
        busy = len(running_camera_pipeline)
    return QueryResourceResponse(
        status=200,
        busy=busy,
        free=len(states) - busy,
        running_camera_pipeline=running_camera_pipeline
    )


async def close_pipeline(request: ClosePipelineRequest):
    success = False
    async with global_consumer_states_lock:
        if request.task_no in global_camera_pipeline:
            pid, gpu_id, id = global_camera_pipeline[request.task_no]
            kill_process = await asyncio.create_subprocess_exec("kill", "-9", f"{pid}", stdout=asyncio.subprocess.PIPE)
            stdout, stderr = await kill_process.communicate()
            await kill_process.wait()
            if stdout is not None:
                global_consumer_states[(gpu_id, id)] = None
                if request.task_no in global_camera_pipeline:
                    del global_camera_pipeline[request.task_no]
                success = True
                # pid 测试
                print(f"kill id is:{pid}")

    response = ClosePipelineResponse(
        status=200 if success else 400,
        task_no=request.task_no
    )
    return response


async def handler_ws_request(raw_data: dict):
    request: WebSocketRequest = WebSocketRequest.parse_obj(raw_data)
    data = request.data
    print(f">>> {raw_data}")
    response = None
    
    if request.action == "AI_TASK":
        response = await create_pipeline(CreatePipelineRequest.parse_obj(data))
    elif request.action == "AI_COMPLETE":
        response = await close_pipeline(ClosePipelineRequest.parse_obj(data))
        response = None
        return
    elif request.action == "AI_QUERY":
        response = await query_resource()
    else:
        response = {"action": "pong", "status": 200}

    if response is None:
        return None

    warpper = WebSocketResponse(
        action=request.action,
        status=0 if (response.status == 200 or response.status == 0) else int(response.status),
        message="Success" if (response.status == 200 or response.status == 0) else "Fail",
        data={}
    )
    response_dict = warpper if isinstance(warpper, dict) else warpper.dict()
    print(f"<<< {response_dict}")
    return response_dict


async def ws_pipeline_endpoint():
    async for websocket in websockets.connect(WEBSOCKET_ADDRESS):
        websocket: WebSocketClientProtocol = websocket
        await manager.connect(websocket)
        try:
            async for message in websocket:
                message = json.loads(message)
                if 'action' not in message:
                    continue
                if message['action'] in ['ping', 'pong']:
                    continue
                response = await handler_ws_request(message)
                if response is not None:
                    await manager.send_personal_message(response, websocket)
        except websockets.ConnectionClosed:
            await manager.disconnect(websocket)
        except Exception as e:
            print(str(e))
        finally:
            continue


class SchedulerService:
    async def heatbeat(self):
        await manager.ping_pong()

    def start(self):
        self.sch = AsyncIOScheduler()
        self.sch.start()
        self.sch.add_job(self.heatbeat, 'interval', seconds=25, max_instances=100)

    def shutdown(self):
        self.sch.shutdown()

@app.on_event("startup")
async def startup_event():
    global global_sch_srv
    loop = asyncio.get_running_loop()
    for gpu_id, id in global_consumer_states.keys():
        loop.create_task(task_executor(gpu_id, id))
    loop.create_task(ws_pipeline_endpoint())
    global_sch_srv = SchedulerService()
    global_sch_srv.start()

@app.on_event("shutdown")
async def shutdown_event():
    global global_sch_srv
    global_sch_srv.shutdown()
    await manager.disconnect_all()

@app.post("/create_pipeline/")
async def create_pipeline_endpoint(item: CreatePipelineRequest):
    return await create_pipeline(item)


@app.get("/query_resource/")
async def query_resource_endpoint():
    return await query_resource()


@app.post("/close_pipeline")
async def close_pipeline_endpoint(request: ClosePipelineRequest):
    return await close_pipeline(request)
 