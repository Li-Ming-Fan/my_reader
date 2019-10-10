#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:19:42 2019
@author: li-ming-fan
"""

import os
import numpy as np

import time
import logging
import json

import tensorflow as tf
from tensorflow.python.framework import graph_util

from abc import ABCMeta, abstractmethod


"""
This class is meant to be task-agnostic.
"""

#
def get_warmup_and_exp_decayed_lr(settings, global_step):
    """ settings.warmup_steps
        settings.decay_steps
        settings.decay_rate
        settings.staircase
        
        learning_rate_schedule = get_warmup_and_exp_decayed_lr
        self.learning_rate_tensor = self.learning_rate_schedule(self.settings, self.global_step)
    """
    learning_rate = tf.constant(value = settings.learning_rate_base,
                                shape = [], dtype = tf.float32)
        
    if settings.warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(settings.warmup_steps, dtype=tf.int32)
        
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        
        step_surplus = global_steps_int - warmup_steps_int
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   step_surplus,
                                                   settings.decay_steps,
                                                   settings.decay_rate,
                                                   settings.staircase)
        
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = settings.learning_rate_base * warmup_percent_done
        
        learning_rate = tf.cond(global_steps_int < warmup_steps_int,
                                lambda: warmup_learning_rate,
                                lambda: learning_rate)
    #
    else:
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   settings.decay_steps,
                                                   settings.decay_rate,
                                                   settings.staircase)
    #
    return learning_rate
    #
    
def get_adam_optimizer(settings, learning_rate_tensor_or_value):
    """ 
        customized_optimizer = get_adam_optimizer
        self._opt = self.customized_optimizer(self.settings, self.learning_rate_tensor)
        
        grad_and_vars = self._opt.compute_gradients(self.loss_train_tensor)
        self.train_op = self._opt.apply_gradients(grad_and_vars, global_step = self.global_step)
    """
    opt = tf.train.AdamOptimizer(learning_rate_tensor_or_value, beta1 = settings.momentum)
    return opt
    #
    
    
#
class ModelBaseboard(metaclass=ABCMeta):
    """
    """    
    def __init__(self, settings):
        """
        """
        # settings
        self.set_model_settings(settings)
        #
        self.learning_rate_schedule = get_warmup_and_exp_decayed_lr  # None
        self.customized_optimizer = get_adam_optimizer
        #
        
    #
    def set_model_settings(self, settings):
        #
        # session info
        if "log_device" not in settings.__dict__.keys():
            settings.__dict__["log_device"] = False 
        #
        if "soft_placement" not in settings.__dict__.keys():
            settings.__dict__["soft_placement"] = True 
        #
        if "gpu_mem_growth" not in settings.__dict__.keys():
            settings.__dict__["gpu_mem_growth"] = True 
        #
        # params
        if "reg_lambda" not in settings.__dict__.keys():
            settings.__dict__["reg_lambda"] = 0.0
        #
        if "reg_exclusions" not in settings.__dict__.keys():
            settings.__dict__["reg_exclusions"] = ["embedding", "bias", "layer_norm", "LayerNorm"]
        #
        if "grad_clip" not in settings.__dict__.keys():
            settings.__dict__["grad_clip"] = 0.0
        #
        if "saver_num_keep" not in settings.__dict__.keys():
            settings.__dict__["saver_num_keep"] = 5
        #
        """
        for key in settings.__dict__.keys():                 
            self.__dict__[key] = settings.__dict__[key]
        """
        #
        # settings
        self.settings = settings
        self.num_gpu = len(settings.gpu.split(","))
        #
        # logger
        self.create_logger()
        #
        # dict
        info_dict = {}
        for name,value in vars(settings).items():
            if not isinstance(value, (int, float, str, bool, list, dict, tuple)):
                continue
            info_dict[str(name)] = value        
        #
        info_str = json.dumps(info_dict, ensure_ascii=False)
        self.logger.info(info_str)
        #
        
    #
    def create_logger(self, log_path=None):
        """
        """
        #
        try:
            self.close_logger()
        except Exception:
            pass
        #
        # logger
        str_datetime = time.strftime("%Y-%m-%d-%H-%M")       
        if log_path is None:
            self.log_path = os.path.join(self.settings.log_dir, "log_" + str_datetime +".txt")
        else:
            self.log_path = log_path
        #
        with open(self.log_path, 'w', encoding='utf-8'):
            pass
        #
        self.logger = logging.getLogger(self.log_path)  # use log_path as log_name
        self.logger.setLevel(logging.INFO)
        #
        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # self.logger.info('test')
        #
        
    def close_logger(self):
        """
        """
        for item in self.logger.handlers:
            item.close()
        #

    #
    @abstractmethod
    def build_placeholder(self):
        """  input_tensors, label_tensors = self.build_placeholder()
        """
        pass
    
    @abstractmethod
    def build_inference(self, input_tensors):
        """ output_tensors = self.build_inference(input_tensors)
            keep_prob = tf.get_variable("keep_prob", shape=[], dtype=tf.float32,
                                        trainable=False)
        """
        pass
    
    @abstractmethod
    def build_loss(self, output_tensors, label_tensors):
        """ loss_tensors = self.build_loss(output_tensors, label_tensors)
        """
        pass
    
    @abstractmethod
    def set_port_tensors(self):
        """ 
        self.src_seq = input_tensors["src_seq"]
        ...
        
        # self.results_train_one_batch = {}
        # self.results_eval_one_batch = {}
        # self.results_debug_one_batch = {}
        
        #
        # feed_dict = self.make_feed_dict_for_predict(x_batch)
        # outputs = self._sess.run(self.outputs_predict, feed_dict = feed_dict)
        #
        
        """
        pass
    
    @abstractmethod
    def make_feed_dict_for_train(self, batch):
        """ feed to input_tensors and label_tensors
        """
        pass

    
    # one_batch functions
    def run_train_one_batch(self, one_batch):
        """ self.results_train_one_batch, NEED to be defined
        """
        feed_dict = self.make_feed_dict_for_train(one_batch)
        results = self._sess.run(self.results_train_one_batch, feed_dict = feed_dict)
        return results
        
    def run_eval_one_batch(self, one_batch):
        """ self.results_eval_one_batch, NEED to be defined
        """
        feed_dict = self.make_feed_dict_for_train(one_batch)        
        results = self._sess.run(self.results_eval_one_batch, feed_dict = feed_dict)
        return results
        
    def run_debug_one_batch(self, one_batch):
        """ self.results_debug_one_batch, NEED to be defined
        """
        assert self.num_gpu == 1, "debug mode can only be run with single gpu"
        feed_dict = self.make_feed_dict_for_train(one_batch)
        results = self._sess.run(self.results_debug_one_batch, feed_dict = feed_dict)        
        return results
    
    #
    # train and validate
    def prepare_for_train(self, dir_ckpt = None):
        """
        """
        # session info
        self.sess_config = tf.ConfigProto(log_device_placement = self.settings.log_device,
                                          allow_soft_placement = self.settings.soft_placement)
        self.sess_config.gpu_options.allow_growth = self.settings.gpu_mem_growth
        #
        # graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            #            
            # model
            input_tensors, label_tensors = self.build_placeholder()
            #
            vs_str = "vs_gpu" # self.settings.vs_str_multi_gpu
            #
            # single_gpu
            if self.num_gpu == 1:
                with tf.variable_scope(vs_str):
                    output_tensors = self.build_inference(input_tensors)
                    loss_tensors = self.build_loss(output_tensors, label_tensors)
                #
                # tensors
                self.input_tensors = input_tensors
                self.label_tensors = label_tensors
                self.output_tensors = output_tensors
                self.loss_tensors = loss_tensors
                #
            else:
                #
                # multi_gpu
                gpu_batch_split = self.settings.gpu_batch_split
                #
                # split among gpu
                inputs_split = []
                labels_split = []
                #
                for idx in range(self.num_gpu):
                    inputs_split.append( {} )
                    labels_split.append( {} )
                #
                for key, value in input_tensors:
                    tensor_split = tf.split(value, gpu_batch_split, axis = 0)
                    for idx in range(self.num_gpu):
                        inputs_split[idx][key] = tensor_split[idx]
                #
                for key, value in label_tensors:
                    tensor_split = tf.split(value, gpu_batch_split, axis = 0)
                    for idx in range(self.num_gpu):
                        labels_split[idx][key] = tensor_split[idx]
                #
                # model, inference, loss
                outputs_dict = {}
                lossputs_dict = {}
                #
                vs_str = self.vs_str_multi_gpu            
                with tf.variable_scope(vs_str):
                    for gid in range(self.num_gpu):
                        with tf.device("/gpu:%d" % gid), tf.name_scope("bundle_%d" % gid):
                            #
                            output_tensors = self.build_inference(inputs_split[gid])
                            loss_tensors = self.build_loss(output_tensors, labels_split[gid])
                            #
                            tf.get_variable_scope().reuse_variables()
                            #
                            # output_tensors
                            for key, value in output_tensors:
                                if key in outputs_dict:
                                    outputs_dict[key] = outputs_dict[key].append(value)
                                else:
                                    outputs_dict[key] = [value]
                                #
                            #
                            # loss_tensors
                            for key, value in loss_tensors:
                                if key in lossputs_dict:
                                    lossputs_dict[key] = lossputs_dict[key].append(value)
                                else:
                                    lossputs_dict[key] = [value]
                                #
                            #
                #
                # outputs
                for key, value in outputs_dict:
                    if len(value[0].get_shape().as_list()) > 0: # rank >= 1
                        #
                        outputs_dict[key] = tf.concat(value, axis=0)
                        #
                #
                # lossputs
                for key, value in lossputs_dict:
                    if len(value[0].get_shape().as_list()) > 0: # rank >= 1
                        #
                        lossputs_dict[key] = tf.concat(value, axis=0)
                        #
                    elif key == "loss_train": # loss_train
                        value_sum = 0
                        for idx in range(self.num_gpu):
                            value_sum += value * gpu_batch_split[idx]
                        #
                        lossputs_dict[key] = value_sum / self.settings.batch_size
                        #
                #
                # tensors
                self.input_tensors = input_tensors
                self.label_tensors = label_tensors
                self.output_tensors = outputs_dict
                self.loss_tensors = lossputs_dict
                #
            #
            # metric and loss
            # if self.settings.use_metric:
            #     self.metric_tensor = self.loss_tensors["metric"]
            #
            self.loss_train_tensor = self.loss_tensors["loss_train"]
            #
            
            #
            # optimizer
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                                               initializer = tf.constant_initializer(0),
                                               trainable = False)
            #
            if self.learning_rate_schedule is not None:
                self.learning_rate_tensor = self.learning_rate_schedule(self.settings, self.global_step)
            else:
                lr = self.settings.learning_rate_base
                self.learning_rate_tensor = tf.get_variable("learning_rate", shape=[], dtype=tf.float32,
                                                            initializer = tf.constant_initializer(lr),
                                                            trainable = False)
            #
            # optimizer
            # optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM, use_nesterov=True)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)              
            # optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = MOMENTUM)
            #
            optimizer_type = self.settings.optimizer_type
            if optimizer_type == 'sgd':
                self._opt = tf.train.GradientDescentOptimizer(self.learning_rate_tensor)
            elif optimizer_type == 'momentum':
                self._opt = tf.train.MomentumOptimizer(self.learning_rate_tensor, self.settings.momentum, use_nesterov=True)
            elif optimizer_type == 'adagrad':
                self._opt = tf.train.AdagradOptimizer(self.learning_rate_tensor, self.settings.adagrad_init_acc)
            elif optimizer_type == 'adam':
                self._opt = tf.train.AdamOptimizer(self.learning_rate_tensor, beta1 = self.settings.momentum)
            elif optimizer_type == 'customized':
                self._opt = self.customized_optimizer(self.settings, self.learning_rate_tensor)
            else:
                assert False, "NOT supported optimizer_type"
            #
            
            #
            # all trainable vars
            self.trainable_vars = tf.trainable_variables()
            # print(self.trainable_vars)
            #
            # regularization
            def is_excluded(v):
                for item in self.settings.reg_exclusions:
                    if item in v.name: return True
                return False
            #
            if self.settings.reg_lambda > 0.0:
                loss_reg = tf.add_n( [tf.nn.l2_loss(v) for v in self.trainable_vars
                                     if not is_excluded(v)] )
                loss_reg = tf.multiply(loss_reg, self.settings.reg_lambda)
                self.loss_train_tensor = tf.add(self.loss_train_tensor, loss_reg)
            #
            # gradient
            grad_and_vars = self._opt.compute_gradients(self.loss_train_tensor)
            #
            # grad_clip           
            if self.settings.grad_clip > 0.0:
                gradients, variables = zip(*grad_and_vars)
                grads, global_norm = tf.clip_by_global_norm(gradients, self.settings.grad_clip)
                grad_and_vars = zip(grads, variables)
                self.global_norm = global_norm
            #
            # train_op
            self.train_op = self._opt.apply_gradients(grad_and_vars, global_step = self.global_step)
            #
            
            #
            # set port tensors
            self.set_port_tensors()
            #
            
            #                
            # save info
            self._saver = tf.train.Saver(max_to_keep = self.settings.saver_num_keep)
            self._saver_best = tf.train.Saver(max_to_keep = self.settings.saver_num_keep)
            
            # sess
            self._sess = tf.Session(graph=self._graph, config = self.sess_config)
            
            #
            # keep_prob
            vs_prefix =  vs_str + "/"
            self._keep_prob_tensor = self._graph.get_tensor_by_name(vs_prefix + "keep_prob:0")
            #
            # initialize the model
            self._sess.run(tf.global_variables_initializer())
            self.assign_dropout_keep_prob(self.settings.keep_prob)
            # self.assign_learning_rate(self.learning_rate_base)
            
            # params count
            self.num_vars = len(self.trainable_vars)
            str_info = 'graph built, there are %d variables in the model' % self.num_vars
            self.logger.info(str_info)
            print(str_info)
            #
            tf_shapes = [tf.shape(v) for v in self.trainable_vars]
            shapes_v = self._sess.run(tf_shapes)
            params_v = [np.prod(item) for item in shapes_v]
            self.param_num = sum(params_v)
            #
            str_info = 'there are %d parameters in the model' % self.param_num
            self.logger.info(str_info)
            print(str_info)
            #
            print()
            for idx in range(self.num_vars):
                print(self.trainable_vars[idx])
                print(params_v[idx])
            print()
            #
        #
        # load
        # if dir_ckpt is None: dir_ckpt = self.model_dir + '_best'
        if dir_ckpt is not None:
            self.logger.info("ckpt loading when prepare for train")
            self.load_ckpt(dir_ckpt)
        else:
            self.logger.info("ckpt not loading when prepare for train")
        #
        
    #
    # assign
    def assign_dropout_keep_prob(self, keep_prob):
        #
        with self._graph.as_default():
            self._sess.run(tf.assign(self._keep_prob_tensor,
                                     tf.constant(keep_prob, dtype=tf.float32)))
        #
        
    def assign_global_step(self, step):
        #
        with self._graph.as_default():
            self._sess.run(tf.assign(self.global_step, tf.constant(step, dtype=tf.int32)))
        #
        
    def assign_learning_rate(self, lr_value):
        #
        with self._graph.as_default():
            self._sess.run(tf.assign(self.learning_rate_tensor,
                                     tf.constant(lr_value, dtype=tf.float32)))
        #
    
    #
    # save and load
    def save_ckpt_best(self, model_dir, model_name, step):
        #
        self._saver_best.save(self._sess, os.path.join(model_dir, model_name),
                              global_step = step)
        
    def save_ckpt(self, model_dir, model_name, step):
        #
        self._saver.save(self._sess, os.path.join(model_dir, model_name),
                         global_step = step)
    
    def load_ckpt(self, dir_ckpt):
        #
        ckpt = tf.train.get_checkpoint_state(dir_ckpt)        
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
            #
            str_info = 'ckpt loaded from %s' % dir_ckpt
            self.logger.info(str_info)
            print(str_info)
        else:
            str_info = 'loading ckpt failed: ckpt loading from %s' % dir_ckpt
            self.logger.info(str_info)
            print(str_info)
            
    #
    # predict, pb
    @staticmethod
    def load_ckpt_and_save_pb_file(model, dir_ckpt):
        """
        """
        is_train = model.settings.is_train
        num_gpu = model.num_gpu
        #
        model.settings.is_train = False                #
        model.num_gpu = 1                              #
        #
        model.prepare_for_train(dir_ckpt)              # loaded here 
        model.assign_dropout_keep_prob(1.0)
        #
        model.set_port_tensors_for_predict()
        #
        pb_file = os.path.join(dir_ckpt, "model_frozen.pb")
        #
        constant_graph = graph_util.convert_variables_to_constants(
                model._sess, model._sess.graph_def,
                output_node_names = model.pb_outputs_name)
        with tf.gfile.GFile(pb_file, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        #
        str_info = 'pb_file saved: %s' % pb_file
        model.logger.info(str_info)
        #
        model.settings.is_train = is_train
        model.num_gpu = num_gpu
        #
        
    def prepare_for_prediction_with_pb(self, pb_file_path = None):
        """ load pb for prediction
        """
        if pb_file_path is None: pb_file_path = self.settings.pb_file 
        if not os.path.exists(pb_file_path):
            assert False, 'ERROR: %s NOT exists, when prepare_for_prediction()' % pb_file_path
        #
        # session info
        self.sess_config = tf.ConfigProto(log_device_placement = self.settings.log_device,
                                          allow_soft_placement = self.settings.soft_placement)
        self.sess_config.gpu_options.allow_growth = self.settings.gpu_mem_growth
        #
        self._graph = tf.Graph()
        with self._graph.as_default():
            with open(pb_file_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
                #
                print('Graph loaded for prediction')
                #
        #
        self._sess = tf.Session(graph = self._graph, config = self.sess_config)
        self.set_port_tensors_for_predict()
        #
    
    def predict_one_batch_with_pb(self, x_batch):
        """ feed_dict = self.make_feed_dict_for_predict(x_batch)
            outputs = self._sess.run(self.outputs_predict, feed_dict = feed_dict) 
        """
        feed_dict = self.make_feed_dict_for_predict(x_batch)
        outputs = self._sess.run(self.outputs_predict, feed_dict = feed_dict)        
        return outputs
    
    #
    # graph and sess
    def get_model_graph_and_sess(self):
        #
        return self._graph, self._sess
        #

#
if __name__ == '__main__':
    
    pass