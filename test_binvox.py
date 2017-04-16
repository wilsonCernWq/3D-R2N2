import inspect
from multiprocessing import Queue
import os
import sys
import json
from collections import OrderedDict
from lib.config import cfg
import inspect
from multiprocessing import Queue

# Training related functions
from models import load_model
from lib.config import cfg
from lib.solver import Solver
from lib.data_io import category_model_id_pair
from lib.data_process import kill_processes, make_data_processes
from lib.data_process import ReconstructionDataProcess


train_queue = Queue(cfg.QUEUE_SIZE)
val_queue = Queue(cfg.QUEUE_SIZE)

train_path = category_model_id_pair(dataset_portion=cfg.TRAIN.DATASET_PORTION)
#train_path structure: category, model_id
category, model_id = train_path[0]
print(category,model_id)
'''
train_processes = make_data_processes(
   train_queue,
   category_model_id_pair(dataset_portion=cfg.TRAIN.DATASET_PORTION),
   cfg.TRAIN.NUM_WORKER,
   repeat=True)
'''
process = ReconstructionDataProcess(train_queue, train_path[0], repeat=True, train=True)
voxel = process.load_label(category,model_id)
voxel_data = voxel.data
