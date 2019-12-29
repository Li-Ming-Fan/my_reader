# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:12:23 2018

@author: limingfan
"""

from Zeras.model_settings_baseboard import ModelSettingsBaseboard


class ModelSettings(ModelSettingsBaseboard):
    """
    """
    def __init__(self):
        """
        """
        super(ModelSettings, self).__init__()
        
        # model graph
        self.model_tag = None
        self.is_train = None
        self.use_metric_in_graph = False

        # vocab
        self.vocab = None
        self.emb_dim = 64
        self.emb_tune = 1  # 1 for tune, 0 for not
        self.tokens_file = None
        self.emb_file = None
        self.token_min_cnt = 5

        # data hyperp
        self.max_p_num = 5
        self.max_p_len = 500
        self.max_q_len = 60
        self.max_a_len = 200

        # train
        self.gpu_available = "0"          # NOT assign here, specified in args
        self.gpu_batch_split = [12, 20]   # list; if None, batch split evenly
                
        self.num_epochs = 100     
        self.batch_size = 32
        self.batch_size_eval = 1
        self.max_batches_eval = 20000
        
        self.reg_lambda = 0.0  # 0.0, 0.01
        self.grad_clip = 0.0  # 0.0, 5.0, 8.0, 2.0
        self.keep_prob = 0.8  # 1.0, 0.7, 0.5
        self.label_smoothing = 0.01
        
        self.optimizer_type = 'adam_wd'  # adam, momentum, sgd, customized
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.learning_rate_base = 0.001   #
        self.learning_rate_minimum = 0.000001
        self.warmup_steps = 1000
        self.decay_steps = 5000
        self.decay_rate = 0.99
        self.staircase = True
        self.lr_power = 1
        self.lr_cycle = True
        
        self.check_period_batch = 100
        self.valid_period_batch = 100

        # save and log, if not set, default values will be used.
        self.base_dir = './task_dureader'
        # self.model_dir = None
        # self.model_name = None
        # self.model_dir_best = None
        # self.pb_file = None
        # self.log_dir = None
        # self.log_path = None
        #

        
if __name__ == '__main__':
    
    sett = ModelSettings()
    #
    sett.model_tag = 'multidocqa'
    sett.is_train = False
    #
    sett.check_settings()
    #
    sett.close_logger()
    #
