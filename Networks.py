import os
import datetime
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorNetwork(keras.Model):
    def __init__(self, gcn_dims, action_dims, layer_nums, name="Actor", chkpt_dir='tmp'):
        super(ActorNetwork, self).__init__()
        self.gcn_dims = gcn_dims
        self.action_dims = action_dims

        self.model_name = name
        current_time = datetime.datetime.now().strftime("%d-%H%M")
        self.checkpoint_dir = os.path.join(chkpt_dir, current_time+"_"+str(action_dims))
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_ddpg.h5')

        # Depending on the number of layers read in from the outside, the number of graph convolution layers is determined
        self.layer_list = []
        for _ in range(layer_nums):
            self.layer_list.append(Dense(self.gcn_dims, activation='relu'))
        # The last layer is the output layer, so no activation is required
        # The first and last layers are not placed in layer_list because of some special operations
        self.fc1 = Dense(self.gcn_dims, activation='relu')
        self.fc2 = Dense(self.action_dims, activation=None)

    def call(self, adj, state):
        # TODO:
        #  In fact, I don't think our model needs to dropout, but overfitting does point to local optima, which can be compared here
        # dropout is used once before each fully connected layer, but the drop rate depends on whether it is train mode and whether it is input information
        #if train:
        input_drop_rate = 0.1
        drop_rate = 0.5
        #else:
        #input_drop_rate = 0.
        #drop_rate = 0.

        adj = tf.convert_to_tensor(adj, dtype=tf.float32)
        # state = tf.nn.dropout(state, rate=input_drop_rate)
        embedding = self.fc1(state)

        for layer_index, layer in enumerate(self.layer_list):

            embedding = layer(tf.matmul(adj, embedding)) + embedding
            # print("============================================")
        logits = self.fc2(embedding)
        return logits
