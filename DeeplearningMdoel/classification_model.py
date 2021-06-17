import tensorflow as tf
import cv2
from flask import Flask, request
from flask_restx import Resource, Api
import base64
import random
import time
from datetime import datetime

app = Flask (__name__)
api = Api(app)

# flask__ image file post , get

@app.route('/camera/ai', methods=['POST','GET'])
def post_and_get():
    image_data = request.get_json('img')

    
    
    raw_image = image_data['img']
    parse_str = raw_image.split(",")[1]
    


    image_de = base64.b64decode(parse_str)


    
    now_str = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    now_png = now_str + '.png'
    
    filename = now_png
    filePath = "C:/Users/mindow/Desktop/Recycle/IMAGE"
     
    with open(filename, 'wb') as f:
        f.write(image_de)
        print("is done")

 # image data training&test       
    
    dropout_rate = 0.5
    label_depth = 4

    RGB_input = tf.placeholder(tf.float32, shape=[256,256,3], name = 'p1')
    RGB_resize = tf.reshape(RGB_input, [1, 256, 256, 3])
    RGB_crop = tf.image.central_crop(RGB_resize, 0.875)

    # conv1
    RW_conv1 = tf.Variable(tf.truncated_normal([7, 7, 3, 96], stddev=0.05), name="RW_conv1")
    Rb_conv1 = tf.Variable(tf.constant(0.0, shape=[96]), name="Rb_conv1")
    Rh_conv1 = tf.nn.relu(tf.nn.conv2d(RGB_crop, RW_conv1, strides=[1, 2, 2, 1], padding="SAME")+Rb_conv1, name="Rh_conv1")
    Rh_norm1 = tf.nn.local_response_normalization(Rh_conv1, depth_radius=5, bias=2, alpha=0.0001, beta=0.75, name="Rh_norm1")
    Rh_pool1 = tf.nn.max_pool(Rh_norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID",name="Rh_pool1")

    # conv2
    RW_conv2 = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.05), name="RW_conv2")
    Rb_conv2 = tf.Variable(tf.constant(0.0, shape=[256]), name="Rb_conv2")
    Rh_conv2 = tf.nn.relu(tf.nn.conv2d(Rh_pool1, RW_conv2, strides=[1, 2, 2, 1], padding="SAME")+Rb_conv2, name="Rh_conv2")
    Rh_norm2 = tf.nn.local_response_normalization(Rh_conv2, depth_radius=5, bias=2, alpha=0.0001, beta=0.75, name="Rh_norm2")
    Rh_pool2 = tf.nn.max_pool(Rh_norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="Rh_pool2")

    # conv3
    RW_conv3 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.05), name="RW_conv3")
    Rb_conv3 = tf.Variable(tf.constant(0.0, shape=[512]), name="Rb_conv3")
    Rh_conv3 = tf.nn.relu(tf.nn.conv2d(Rh_pool2, RW_conv3, strides=[1, 1, 1, 1], padding="SAME")+Rb_conv3, name="Rh_conv3")

    # conv4
    RW_conv4 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.05), name="RW_conv4")
    Rb_conv4 = tf.Variable(tf.constant(0.0, shape=[512]), name="Rb_conv4")
    Rh_conv4 = tf.nn.relu(tf.nn.conv2d(Rh_conv3, RW_conv4, strides=[1, 1, 1, 1], padding="SAME")+Rb_conv4, name="Rh_conv4")

    # conv5
    RW_conv5 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.05), name="RW_conv5")
    Rb_conv5 = tf.Variable(tf.constant(0.0, shape=[512]), name="Rb_conv5")
    Rh_conv5 = tf.nn.relu(tf.nn.conv2d(Rh_conv4, RW_conv5, strides=[1, 1, 1, 1], padding="SAME")+Rb_conv5, name="Rh_conv5")
    Rh_pool5 = tf.nn.max_pool(Rh_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="Rh_pool5")
    Rh_pool5_flat = tf.reshape(Rh_pool5, [-1, 6*6*512], name="Rh_pool5_flat")

    # fc1
    RW_fc1 = tf.Variable(tf.truncated_normal([6*6*512, 4096], dtype=tf.float32, stddev=0.005 ), name="RW_fc1")
    Rb_fc1 = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), name="Rb_fc1")
    Rh_fc1 = tf.nn.relu(tf.matmul(Rh_pool5_flat, dropout_rate * RW_fc1)+Rb_fc1, name="Rh_fc1")


    # fc2
    RW_fc2 = tf.Variable(tf.truncated_normal([4096, 2048], dtype=tf.float32, stddev=0.005), name="RW_fc2")
    Rb_fc2 = tf.Variable(tf.constant(0.0, shape=[2048]), name="Rb_fc2")
    Rh_fc2 = tf.nn.relu(tf.matmul(Rh_fc1, dropout_rate * RW_fc2)+Rb_fc2, name="Rh_fc2")


    # fc3
    RW_fc3 = tf.Variable(tf.truncated_normal([2048, label_depth], dtype=tf.float32, stddev=0.005), name="RW_fc3")
    Rb_fc3 = tf.Variable(tf.constant(0.0, shape=[label_depth]),name="Rb_fc3")
    Ry_out = tf.nn.softmax(tf.matmul(Rh_fc2, RW_fc3) + Rb_fc3, name="Ry_out")

    Output = tf.argmax(Ry_out, 1)+1

    saver = tf.train.Saver([RW_conv1, Rb_conv1, RW_conv2, Rb_conv2, RW_conv3, Rb_conv3, RW_conv4, Rb_conv4, RW_conv5, Rb_conv5, RW_fc1, Rb_fc1, RW_fc2, Rb_fc2, RW_fc3, Rb_fc3], max_to_keep=81)


# classification model starting point

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=sess, coord=coord)


        sess.run(tf.global_variables_initializer())

        saver.restore(sess, 'Weights')



        frame = cv2.imread(now_png)
        image = cv2.resize(frame, dsize=(256, 256), interpolation=cv2.INTER_AREA)

        Result = sess.run(Output, feed_dict={RGB_input:image})
        
        


        if Result == [1]:
            result_value = "Plastic Bottle"
            print("Plastic Bottle")

        elif Result == [2]:
            result_value = "Note"
            print("Note")

        elif Result == [3]:
            result_value = "Butain"
            print("Butain")
        
        else :
            result_value = "error! 사진을 확인해주세요!"
        
        


#         #test 코드, 이미지가 제대로 출력되는 확인하기 위한 코드(주석처리 가능)
#         cv2.imshow('test', image) 
#         cv2.waitKey()
#         cv2.destroyAllWindows()

        coord.request_stop()
        coord.join(thread)
        
        
    return {"result" : result_value }


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)




