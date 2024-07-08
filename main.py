import os
import time
import sys
import math
import numpy as np
import tensorflow as tf

from minPE import countMinPE, loadData
from Agent import Agent
from environment_routing import Environment
from graph_routing import Graph
from config import get_config
from dataGenerator import DataGenerator

"""
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
np.set_printoptions(threshold=sys.maxsize)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      tf.config.experimental.set_virtual_device_configuration(
              gpu,
              [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 0.5)])
"""
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

def main():
    # Parameters are defined through command line input, which makes it easy to modify parameters
    config = get_config()
    # Read the query and target graph paths
    source_file_path = config.src_file_path
    # Read the number of batches, the maximum number of cycles, and the storage path of the model
    # Many of the following are added with an int function, in order to prevent the input float, causing unnecessary trouble
    batch_size = int(config.batch_size)
    max_iteration = int(config.max_iteration)
    head_nums = int(config.head_nums)
    ckpt_dir = config.ckpt_dir

    actor_lr = config.actor_lr
    # When the bool value is read by the parameter, it is read as string, so there should be some variation when using it
    load_model = config.load_model
    # Map width, softmax temperature
    pea_width = config.pea_width
    temperature = config.temperature
    beta = config.beta
    layer_nums = config.layer_nums
    C = config.c
    max_LRF = config.max_LRF
    max_GRF = config.max_GRF
    max_memory = config.max_memory
    memory_mode = config.memory_mode
    reward_mode = config.reward_mode
    min_ii = config.mii
    # Read the dimensions of the graph convolutional network
    hidden_dims = int(config.hidden_dims)

    # Create a root directory to store the weights
    ckpt_dir = os.path.join(os.getcwd(), ckpt_dir)
    if not os.path.exists(path=ckpt_dir):
        os.mkdir(ckpt_dir)

    # TODO Manually represent the required Action dimension
    if memory_mode:
        action_dims = pea_width ** 2 + 5
    else:
        action_dims = pea_width ** 2 + 2

    start_time = time.time()
    raw_data = loadData(source_file_path)

    embedding = countMinPE(data=raw_data, II=min_ii)
    source_graph = Graph(origin_embedding=embedding, ii=min_ii)
    # Make dataset
    generator = DataGenerator(graph=source_graph)
    total_adj, total_embedding, total_graph, total_net_input, total_dict = generator.generate()
    # init Agent
    agent = Agent(total_adj=total_adj, total_embedding=total_embedding, total_graph=total_graph, layer_nums=layer_nums,
                          pea_width=pea_width, actor_lr=actor_lr, batch_size=batch_size, hidden_dims=hidden_dims,                                    
                          beta=beta, ii=min_ii, total_dict=total_dict, temperature=10, C=5, max_LRF=max_LRF,
                          max_GRF=max_GRF, action_dims=action_dims, memory_mode=True, max_memory=max_memory, reward_mode=reward_mode,
                          transfer_learning=False, total_net_input=total_net_input, source_graph=source_graph)

    
    for episode in range(max_iteration):
        train = True
        load_model = False


        result_flag = agent.learn(episode=episode, load_model=load_model)

        if result_flag:
            end_time = time.time()
            print("total time:",(end_time-start_time))
            break


if __name__ == '__main__':
    main()
