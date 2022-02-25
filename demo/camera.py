import cv2
from flask import Flask, render_template, Response
from demo_skeleton import main
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


def gen():
    #"C:\Users\pjw\Desktop\a_ymDZuv0U2pdh1590978635.mp4"
    # video_path = 'E:/GraduationProject/dataset/video/video1.mp4'
    # video_path = 'C:/Users/pjw/Desktop/a_ymDZuv0U2pdh1590978635.mp4'
    # vid = cv2.VideoCapture(video_path)
    vis_frames = main()
    while True:
        # return_value, frame = vid.read()
        for frame in vis_frames:
            image = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
