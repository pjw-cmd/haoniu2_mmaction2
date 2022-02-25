import cv2
import queue
import time
import threading
from flask import Flask,make_response,json,jsonify,render_template,Response
from demo_skeleton import main,main1
import os.path as osp
import os
import mmcv
from mmcv import DictAction
import numpy as np
import argparse
from mmaction.apis import inference_recognizer, init_recognizer
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (init_pose_model, inference_top_down_pose_model,
                             vis_pose_result)
app = Flask(__name__)


q = queue.Queue()

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth'),
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/skeleton/label_map_ntu120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args

args = parse_args()
config = mmcv.Config.fromfile(args.config)

model = init_recognizer(config, args.checkpoint, args.device)
pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)
detection_model = init_detector(args.det_config, args.det_checkpoint, args.device)
label_map = [x.strip() for x in open(args.label_map).readlines()]



def Receive():
    print("start Reveive")
    user, pwd, ip, channel = "admin", "pjw123456", "10.1.16.80", 1
    cap_path = "rtsp://%s:%s@%s/h264/ch%s/main/av_stream" % (user, pwd, ip, channel)
    cap = cv2.VideoCapture(cap_path)
    # "C:\Users\pjw\Desktop\mda-kgutcmw87usyb3ik.mp4"
    # cap = cv2.VideoCapture("C:/Users/pjw/Desktop/mda-kgutcmw87usyb3ik.mp4")
    ret, frame = cap.read()
    q.put(frame)
    while True:
        ret, frame = cap.read()
        q.put(frame)
        q.get() if q.qsize() > 1 else time.sleep(0.01)
def Display():
    print("Start Displaying")
    # cv2.namedWindow("123", flags=cv2.WINDOW_FREERATIO)
    target_dir = osp.join('./tmp')
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    frame_paths = []

    pose_results = []
    frames = []
    new_h, new_w = None, None
    cnt = 0
    action = "123"
    while True:
        if q.empty()!=True:
            frame = q.get()
            # cv2.imshow("frame1", frame)
            if new_h is None:
                h, w, _ = frame.shape
                new_w, new_h = mmcv.rescale_size((w, h), (480, np.Inf))

            frame = mmcv.imresize(frame, (new_w, new_h))

            # det_results = detection_inference(args, frame_paths, detection_model)

            frame_path = frame_tmpl.format(cnt + 1)
            frame_paths.append(frame_path)
            cv2.imwrite(frame_path, frame)
            cnt += 1
                # frame = q.get()
            pose_result,vis_frame = main(frame_path,args,config,model,pose_model,detection_model,label_map)
            cv2.putText(vis_frame, action, (30, 60), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)
            frames.append(vis_frame)
            pose_results.append(pose_result)
            if cnt != 50:
                image = cv2.imencode('.jpg', vis_frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

            if cnt == 50:
                vis_frame,action_label = main1(frame_paths,frames,pose_results,model,label_map,pose_model,vis_frame)
                action = action_label
                pose_results.clear()
                frame_paths.clear()
                frames.clear()
                cnt = 0
                # for frame in vis_frames:
                image = cv2.imencode('.jpg', vis_frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

            # frame = vis_frames.pop()
            # image = cv2.imencode('.jpg', frame)[1].tobytes()
            # for frame in vis_frames:
            #     image = cv2.imencode('.jpg', frame)[1].tobytes()
            #     yield (b'--frame\r\n'
            #            b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

# def Display1():
#     print("Start Displaying")
#     while vis_frames.empty()!=True:
#         # return_value, frame = vid.read()
#         print(2)
#         vis_frame = vis_frames.get()
#         for frame in vis_frame:
#             image = cv2.imencode('.jpg', frame)[1].tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            # # 加入对帧的处理算法，返回帧
            # ret, jpeg = cv2.imencode('.jpg',frame)
            # # cv2.imshow("123", frame)
            # frame = jpeg.tobytes()
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')

@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(Display(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    # p3 = threading.Thread(target=Display1)
    p1.start()
    p2.start()
    # p3.start()
    app.run(host='0.0.0.0', debug=True, port=5000)

