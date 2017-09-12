# -*- encoding:utf8 -*-
import tensorflow as tf
'''
多模型示例
'''

g1 = tf.Graph() # 加载到Session 1的graph
g2 = tf.Graph() # 加载到Session 2的graph

sess1 = tf.Session(graph=g1) # Session1
sess2 = tf.Session(graph=g2) # Session2

# 加载第一个模型
with sess1.as_default(): # 1
    with g1.as_default():
        # tf.global_variables_initializer().run()
        # model_saver = tf.train.Saver(tf.global_variables())
        model_saver  = tf.train.import_meta_graph("model/1/model.ckpt-1000000.meta")
        model_ckpt = tf.train.get_checkpoint_state("model/1/")
        print(model_ckpt)
        print(model_ckpt.model_checkpoint_path)
        model_saver.restore(sess1, model_ckpt.model_checkpoint_path)
# 加载第二个模型
with sess2.as_default():  # 2
    with g2.as_default():  
        # tf.global_variables_initializer().run()
        # model_saver = tf.train.Saver(tf.global_variables())
        model_saver  = tf.train.import_meta_graph("model/2/model.ckpt-993077.meta")
        model_ckpt = tf.train.get_checkpoint_state("model/2/")
        model_saver.restore(sess2, model_ckpt.model_checkpoint_path)

'''
# 使用的时候
with sess1.as_default():
    with sess1.graph.as_default():  # 2
        ...

with sess2.as_default():
    with sess2.graph.as_default():
        ...
'''
# 关闭sess
sess1.close()
sess2.close()