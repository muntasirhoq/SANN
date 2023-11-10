# -*- coding: utf-8 -*-
"""
Created on Wed May 11 23:07:03 2022

@author: Muntasir Hoq
"""
class Config:
    def __init__(self):
        self.embedding_size = 256
        self.node_embedding_size = 256
        self.path_embedding_size = 256
        self.max_path_length = 100
        self.max_paths = 100
        self.num_classes = 2
        self.batch_size = 128
        self.epoch = 200

        self.assignmentID = 439
        self.problemID = 13
        self.depth = 4
        self.leaf_flag = 1 # 1: with leaves; 0: leaves pruned
        self.test_train_flag = "Test"