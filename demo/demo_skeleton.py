# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil

import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction

from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.utils import import_module_error_func



try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import (init_pose_model, inference_top_down_pose_model,
                             vis_pose_result)
except (ImportError, ModuleNotFoundError):

    @import_module_error_func('mmdet')
    def inference_detector(*args, **kwargs):
        pass

    @import_module_error_func('mmdet')
    def init_detector(*args, **kwargs):
        pass

    @import_module_error_func('mmpose')
    def init_pose_model(*args, **kwargs):
        pass

    @import_module_error_func('mmpose')
    def inference_top_down_pose_model(*args, **kwargs):
        pass

    @import_module_error_func('mmpose')
    def vis_pose_result(*args, **kwargs):
        pass


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

# import time
# from functools import wraps
# def fn_timer(function):
#     @wraps(function)
#     def function_timer(*args, **kwargs):
#         t0 = time.time()
#         result = function(*args, **kwargs)
#         t1 = time.time()
#         print("Total time running %s: %s seconds" % (function.__name__, str(t1-t0)))
#         return result
#     return function_timer

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


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')

    # 下面两行我新增的代码
    # user, pwd, ip, channel = "admin", "pjw123456", "10.1.16.80", 1
    # video_path = "rtsp://%s:%s@%s/h264/ch%s/main/av_stream" % (user, pwd, ip, channel)

    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    flag1 = 50
    while flag1:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
        flag1 = flag1-1

    return frame_paths, frames


def detection_inference(args, frame_paths,detection_model):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    # model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert detection_model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    # print('Performing Human Detection for each frame')
    # prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(detection_model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        # prog_bar.update()
    return results
# @fn_timer
def detection_inference1(args, frame_path,detection_model):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    # model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert detection_model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    # results = []
    # print('Performing Human Detection for each frame')
    # prog_bar = mmcv.ProgressBar(len(frame_paths))
    # for frame_path in frame_paths:
    result = inference_detector(detection_model, frame_path)
    # We only keep human detections with score larger than det_score_thr
    result = result[0][result[0][:, 4] >= args.det_score_thr]
    # results.append(result)
        # prog_bar.update()
    return result


def pose_inference(args, frame_paths, det_results,pose_model):
    # model = init_pose_model(args.pose_config, args.pose_checkpoint,
    #                         args.device)
    ret = []
    # print('Performing Human Pose Estimation for each frame')
    # prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(pose_model, f, d, format='xyxy')[0]
        ret.append(pose)
        # prog_bar.update()
    return ret
# @fn_timer
def pose_inference1(args, frame_path, det_result,pose_model):
    # model = init_pose_model(args.pose_config, args.pose_checkpoint,
    #                         args.device)
    # ret = []
    # print('Performing Human Pose Estimation for each frame')
    # prog_bar = mmcv.ProgressBar(len(frame_paths))
    # for f, d in zip(frame_paths, det_results):
        # Align input format
    d = [dict(bbox=x) for x in list(det_result)]
    pose = inference_top_down_pose_model(pose_model, frame_path, d, format='xyxy')[0]
    # ret.append(pose)
        # prog_bar.update()
    return pose



def main(frame_path,args,config,model,pose_model,detection_model,label_map):
    # args = parse_args()

    # frame_paths, frames = frame_extraction(args.video,
    #                                                 args.short_side)

    # Get clip_len, frame_interval and calculate center index of each clip
    # config = mmcv.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    for component in config.data.test.pipeline:
        if component['type'] == 'PoseNormalize':
            component['mean'] = (w // 2, h // 2, .5)
            component['max_value'] = (w, h, 1.)

    # model = init_recognizer(config, args.checkpoint, args.device)

    # Load label_map
    # label_map = [x.strip() for x in open(args.label_map).readlines()]

    # Get Human detection results
    det_result = detection_inference1(args, frame_path,detection_model)
    torch.cuda.empty_cache()

    pose_result = pose_inference1(args, frame_path, det_result,pose_model)
    torch.cuda.empty_cache()

    vis_frame = vis_pose_result(pose_model, frame_path, pose_result)


    return pose_result,vis_frame
def main1(frame_paths,frames,pose_results,model,label_map,pose_model,vis_frame):

    num_frame = len(frame_paths)
    h, w,_ = frames[0].shape

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    num_person = max([len(x) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :2]
            keypoint_score[j, i] = pose[:, 2]
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    results = inference_recognizer(model, fake_anno)

    action_label = label_map[results[0][0]]

    # pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
    #                              args.device)
    # vis_frames = [
    #     vis_pose_result(pose_model, frame_paths[i], pose_results[i])
    #     for i in range(num_frame)
    # ]
    # for frame in frames:
    cv2.putText(vis_frame, action_label, (30, 60), FONTFACE, FONTSCALE,
                FONTCOLOR, THICKNESS, LINETYPE)

    # vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    # vid.write_videofile(args.out_filename, remove_temp=True)

    # tmp_frame_dir = osp.dirname(frame_paths[0])
    # shutil.rmtree(tmp_frame_dir)

    return vis_frame,action_label


if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
    app.run(host='0.0.0.0', debug=True, port=5000)
    main()



