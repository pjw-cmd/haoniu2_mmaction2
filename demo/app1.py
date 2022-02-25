import cv2
import queue
import time
import threading
from flask import Flask,make_response,json,jsonify,render_template,Response
from demo_video_structuralize import main
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



def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--rgb-stdet-config',
        default=('configs/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py'),
        help='rgb-based spatio temporal detection config file path')
    parser.add_argument(
        '--rgb-stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='rgb-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--skeleton-stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'posec3d_ava.pth'),
        help='skeleton-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/'
                 'faster_rcnn/faster_rcnn_r50_fpn_2x_coco/'
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
        '--skeleton-config',
        default='configs/skeleton/posec3d/'
        'slowonly_r50_u48_240e_ntu120_xsub_keypoint.py',
        help='skeleton-based action recognition config file path')
    parser.add_argument(
        '--skeleton-checkpoint',
        default='https://download.openmmlab.com/mmaction/skeleton/posec3d/'
        'posec3d_k400.pth',
        help='skeleton-based action recognition checkpoint file/url')
    parser.add_argument(
        '--rgb-config',
        default='configs/recognition/tsn/'
        'tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py',
        help='rgb-based action recognition config file path')
    parser.add_argument(
        '--rgb-checkpoint',
        default='https://download.openmmlab.com/mmaction/recognition/'
        'tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/'
        'tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth',
        help='rgb-based action recognition checkpoint file/url')
    parser.add_argument(
        '--use-skeleton-stdet',
        action='store_true',
        help='use skeleton-based spatio temporal detection method')
    parser.add_argument(
        '--use-skeleton-recog',
        action='store_true',
        help='use skeleton-based action recognition method')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.4,
        help='the threshold of action prediction score')
    parser.add_argument(
        '--video',
        default='demo/test_video_structuralize.mp4',
        help='video file/url')
    parser.add_argument(
        '--label-map-stdet',
        default='tools/data/ava/label_map.txt',
        help='label map file for spatio-temporal action detection')
    parser.add_argument(
        '--label-map',
        default='tools/data/kinetics/label_map_k400.txt',
        help='label map file for action recognition')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--out-filename',
        default='demo/test_stdet_recognition_output.mp4',
        help='output filename')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a spatio-temporal detection prediction per n frames')
    parser.add_argument(
        '--output-stepsize',
        default=1,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--output-fps',
        default=24,
        type=int,
        help='the fps of demo video output')
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
    while True:
        flag1 = 50
        if q.empty()!=True:
            frame = q.get()
            # cv2.imshow("frame1", frame)
            target_dir = osp.join('./tmp')
            os.makedirs(target_dir, exist_ok=True)
            # Should be able to handle videos up to several hours
            frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
            frames = []
            frame_paths = []
            # flag, frame = vid.read()
            cnt = 0
            new_h, new_w = None, None
            while flag1:
                frames.append(frame)
                frame_path = frame_tmpl.format(cnt + 1)
                frame_paths.append(frame_path)
                cv2.imwrite(frame_path, frame)
                cnt += 1
                frame = q.get()
                flag1 = flag1 - 1

            vis_frames = main(frame_paths, frames,args)
            # frame = vis_frames.pop()
            # image = cv2.imencode('.jpg', frame)[1].tobytes()
            for frame in vis_frames:
                image = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

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
    app.run(host='0.0.0.0', debug=True, port=5050)

