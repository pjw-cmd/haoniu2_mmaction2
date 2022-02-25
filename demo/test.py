import cv2
import uuid


def test_download():
    # cap = cv2.VideoCapture("rtsp://3.84.6.190/vod/mp4:BigBuckBunny_115k.mov")
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("rtsp://admin:12345@192.168.1.64/main/Channels/1")
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_s = cap.get(5)
    print(frame_s)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ret, frame = cap.read()
    time_frame = frame_s  * 5  # 设置保存时间为五分钟一保存
    num = 0
    while ret:
        if num == 0:
            filename = str(uuid.uuid4()) + ".mp4"
            video_writer = cv2.VideoWriter(filename, fourcc, frame_s, size, True)  # 参数：视频文件名，格式，每秒帧数，宽高，是否灰度
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        img = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
        video_writer.write(frame)
        num = num + 1
        if num == time_frame:
            video_writer.release()
            num = 0
            break
    # video_writer.release()
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    test_download()