import os
import cv2
import gc
from multiprocessing import Process, Manager
import cv2

# 向共享缓冲栈中写入数据:
def write(stack, top: int) -> None:
    """
    :param cam: 摄像头参数
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    user, pwd, ip, channel = "admin", "pjw123456", "10.1.16.80", 1
    cap_path = "rtsp://%s:%s@%s/h264/ch%s/main/av_stream" % (user, pwd, ip, channel)
    #cap = cv2.VideoCapture(cap_path)
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        if _:
            stack.append(img)
            # 每到一定容量清空一次缓冲栈
            # 利用gc库，手动清理内存垃圾，防止内存溢出
            if len(stack) >= top:
                del stack[:]
                gc.collect()


# 在缓冲栈中读取数据:
def read(stack) -> None:
    print('Process to read: %s' % os.getpid())
    while True:
        if len(stack) != 0:
            value = stack.pop()
            # 对获取的视频帧分辨率重处理
            # img_new = img_resize(value)
            # 使用yolo模型处理视频帧
            # yolo_img = yolo_deal(value)
            # 显示处理后的视频帧
            cv2.imshow("img", value)
            # 将处理的视频帧存放在文件夹里
            # save_img(yolo_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    q = Manager().list()
    pw = Process(target=write, args=(q, 100))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pr结束:
    pr.join()

    # pw进程里是死循环，无法等待其结束，只能强行终止:
    pw.terminate()
