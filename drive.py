#import libraries
import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
from tensorflow.keras.losses import MeanSquaredError

#initialize socketio and Flask server 
sio = socketio.Server()

app = Flask(__name__) #'__main__'

speed_limit = 10  

#apply image preprocessing for prediction
def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

#connect with udacity simulator
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


#take streeting angle from predicition model and keep sending it to the simulator
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


#load up the image and predict the steering angle
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed / speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

#connect with simulator
if __name__ == '__main__':
    model = load_model('model/model2.h5', custom_objects={'mse': MeanSquaredError()})
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)




