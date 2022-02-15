import time
import os
import pickle
import numpy as np
import torch


class IOManager():
    def __init__(self, learner):
        self.learner = learner
        self.config = learner.config

        self.result_path = None
        self.log = None

    def initialize(self):
        self.result_path = self.config.path_save + '/' + self.config.learn_name + '/' + str(self.config.kmer) + "mer"
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        # 生成一份pkl的文件
        with open(self.result_path + '/config.pkl', 'wb') as file:
            pickle.dump(self.config, file)
        self.log = LOG(self.result_path)

        # 生成一份txt文件
        with open(self.result_path + '/config.txt', 'w') as f:
            for key, value in self.config.__dict__.items():
                key_value_pair = '{}: {}'.format(key, value)
                f.write(key_value_pair + '\r\n')

    # 保存最佳模型
    def save_model_dict(self, model_dict, save_prefix, metric_name, metric_value):
        filename = '{}, {}[{:.3f}].pt'.format(save_prefix, metric_name, metric_value)
        # filename = '{},ACC{}.pt'.format(save_prefix, metric_name, metric_value)
        save_path_pt = os.path.join(self.result_path, filename)
        torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)

# 打印日志文件
class LOG():
    def __init__(self, root_path):
        log_path = root_path + '/log'
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.log = open(log_path + '/%s.txt' % time.strftime("%Y_%m_%d_%I_%M_%S"), 'w+')
        log_fils = os.listdir(log_path)
        log_fils.sort()
        if len(log_fils) > 200:
            print('log file >200, delete old file', log_fils.pop(0), file=self.log)

    def Info(self, *data):
        msg = time.strftime("%Y-%m-%d_%I:%M:%S") + " INFO: "
        for info in data:
            if type(info) == int:
                msg = msg + str(info)
            else:
                msg = msg + str(info)
        print(msg)
        # print >>self.log,msg
        print(msg, file=self.log)

    def Warn(self, *data):
        msg = time.strftime("%Y-%M-%d_%I:%M:%S") + " WARN: "
        for info in data:
            if type(info) == int:
                msg = msg + str(info)
            else:
                msg = msg + info
        print(msg)
        # print >> self.log, msg
        print(msg, file=self.log)

    def Error(self, *data):
        msg = time.strftime("%Y-%M-%d_%I:%M:%S") + " ERROR: "
        for info in data:
            if type(info) == int:
                msg = msg + str(info)
            else:
                msg = msg + info
        print(msg)
        # print >> self.log, msg
        print(msg, file=self.log)
