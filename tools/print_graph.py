

import os
import sys
sys.path.append(os.getcwd())


import argparse
import os.path as ops
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.python import pywrap_tensorflow

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')





def savepb(weights_path):


    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        writer = tf.summary.FileWriter('logs', sess.graph)  # 保存图结构


    output_node_names = "LaneNet/bisenetv2_backend/instance_seg/pix_embedding_conv/pix_embedding_conv,LaneNet/bisenetv2_backend/binary_seg/ArgMax"

    # output_node_names = "lanenet_model/bisenetv2_backend/instance_seg/pix_embedding_conv/pix_embedding_conv"
    # # output_node_names = 'lanenet_model/vgg_backend/instance_seg/pix_embedding_conv/pix_embedding_conv'
    input_graph_def = sess.graph.as_graph_def()
    # output_node_names="train_IteratorGetNext"
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        output_node_names.split(","))
    graph_io.write_graph(output_graph_def, './', 'lanenet.pb', as_text=False)

    # define saver
    # saver = tf.train.Saver(variables_to_restore)

    # with sess.as_default():
    #     saver.restore(sess=sess, save_path=weights_path)

    #     tf.import_graph_def(tf.get_default_graph())
    #     [print(n.name) for n in tf.get_default_graph().as_graph_def().node]




    # # 打印参数
    # reader=pywrap_tensorflow.NewCheckpointReader(weights_path)
    # var_to_shape_map=reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print('tensor_name: ',key)


    # # 打印节点
    # # read node name way 1
    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph(weights_path + '.meta', clear_devices=True)
    #     graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    #     node_list = [n.name for n in graph_def.node]
    #     for node in node_list:
    #         print("node_name", node)


    sess.close()



if __name__ == '__main__':
    weights_path = "./model/tusimple_lanenet.ckpt"
    savepb(weights_path)





