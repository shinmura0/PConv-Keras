import cv2
import time
import os
import numpy as np
#from libs.pconv_model import PConvUnet
from libs.pconv_layer import PConv2D
import keras
from keras.models import model_from_json
from keras import backend as K

m_input_size, m_input_size = 256, 256

print("Model loading...")
model = keras.models.load_model('model/model.h5', compile=False, custom_objects={'PConv2D': PConv2D})

def main():
    camera_width =  352
    camera_height = 288
    fps = ""
    elapsedTime = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    time.sleep(1)

    while cap.isOpened():
        t1 = time.time()

        ret, image = cap.read()
        if not ret:
            break

        image = image[16:272, 48:304]
            
        # Mask
        img = np.zeros(image.shape, np.uint8)
        cv2.rectangle(img, (70, 70), (130, 130), (1, 1, 1), thickness=-1)
        mask = 1-img

        # Image + mask
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (m_input_size, m_input_size)) / 255
        img[mask==0] = 1
        predict_img = model.predict([np.expand_dims(img, axis=0), np.expand_dims(mask, axis=0)])

        output = predict_img.reshape(image.shape)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.putText(output, fps, (camera_width-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Result", output)
            
        # FPS
        elapsedTime = time.time() - t1
        fps = "{:.0f} FPS".format(1/elapsedTime)
        print(fps)

        # quit
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
