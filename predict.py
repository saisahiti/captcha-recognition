#! /usr/bin/python
# _*_ coding: utf-8 _*_


import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
import cv2
import os
import mymodel
import time
from constant import width, height, char_num, characters, classes
from utils import get_captcha

def predict_image(captcha):
    """
    predict captcha of single image

    :captcha: captcha img path or captcha img ndarray data
    :return: a string containing a captcha
    """
    tf.reset_default_graph() 
    if isinstance(captcha, np.ndarray):
        img = captcha
    elif isinstance(captcha, str):
        if os.path.exists(captcha):
            img = cv2.imread(captcha, 0)  # (100, 120)
        else:
            raise FileNotFoundError('captcha not exists')
    else:
        raise ValueError('the captcha param should be '
                         'a path of img or ndarray img data')

    img[img < 193] = 0
    img[img >= 193] = 1
    img = np.reshape(img, [1, img.shape[0], img.shape[1], 1])
    x = tf.placeholder(tf.float32, [None, height, width, 1])
    y_conv, _ = mymodel.captcha_model(x, keep_prob=1, trainable=False)
    predict = tf.argmax(tf.reshape(y_conv, [-1, char_num, classes]), 2)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
	# uncomment codes below if need to validate mutilple imgs in the same program
    # tf.reset_default_graph()
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state('./model_data')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('oh, bad model, please check')
    pre_list = sess.run(predict, feed_dict={x: img})
    return get_captcha(pre_list, characters)


# if __name__ == '__main__':
    # start = time.time()
    # # infer 45-55ms
    # # load model 195-210ms
    # url='/home/bavya/dataset/images/char-4-epoch-1/test/'
    # images=['bdip_8393caeb-25f5-4d8b-8ba8-434aa718655d.png','btqy_664630aa-368a-4912-a691-6a98d38c9784.png','aymt_c9daf9d3-38ee-4e6f-ae73-292f2a7d54dc.png','aryd_e29b0b46-c3c2-476d-b281-6a16b4960641.png']
    # s = predict_image(url+images[0])
    # print("expected output : ",images[0][0:4])
    # print("actual output:  ",s)
    # s = predict_image(url+images[1])
    # print("expected output : ",images[1][0:4])
    # print("actual output:  ",s)
    # s = predict_image(url+images[2])
    # print("expected output : ",images[2][0:4])
    # print("actual output:  ",s)
    # s = predict_image(url+images[3])
    # print("expected output : ",images[3][0:4])
    # print("actual output:  ",s)
    # print('finished in %d ms' % ((time.time() - start) * 1000.))
