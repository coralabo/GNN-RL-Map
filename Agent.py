import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from scipy import sparse
from Networks import ActorNetwork
from environment_routing import Environment


class Agent:
    def __init__(self, action_dims, max_memory, memory_mode, reward_mode, max_GRF, max_LRF, total_adj, total_embedding, total_graph,
                 C, temperature, source_graph, total_dict, total_net_input, ii, transfer_learning=False, beta=0.2, layer_nums=7, pea_width=4, chkpt_dir="tmp/ddpg",
                 actor_lr=1e-3, hidden_dims=32, batch_size=64):
        self.baseline = None
        self.graph = source_graph
        self.batch_size = batch_size
        self.C = C
        self.ii = ii
        self.temperature = temperature
        self.environment = Environment(C=C, temperature=temperature, total_adj=total_adj, max_LRF=max_LRF, total_net_input=total_net_input,
                                       total_embedding=total_embedding, total_graph=total_graph, action_dims=action_dims,
                                       total_dict=total_dict, pea_width=pea_width, beta=beta, ii=ii, max_GRF=max_GRF,
                                       memory_mode=memory_mode, max_memory=max_memory, reward_mode=reward_mode)

        self.actor = ActorNetwork(gcn_dims=hidden_dims, action_dims=action_dims,
                                  name="actor", chkpt_dir=chkpt_dir, layer_nums=layer_nums)
        if not transfer_learning:
            # If it is not transfer learning, then the learning rate for all layers is actor_lr
            self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        else:
            # If it is transfer learning, the learning rate of the previous layers is one-tenth of the original, except that the learning rate of the last layer is actor_lr
            optimizers = [
                        Adam(learning_rate=actor_lr/10),
                        Adam(learning_rate=actor_lr)
                        ]
            optimizers_and_layers = [(optimizers[0], self.actor.layers[0:-1]), (optimizers[1], self.actor.layers[-1])]
            #optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
            #self.actor.compile(optimizer=optimizer, loss="mse")
                                                                                                                    
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def save_models(self):
        print(".... saving models ....")
        tf.saved_model.save(self.actor, self.actor.checkpoint_dir)

    def load_part_models(self):
        print(".... loading part models ....")
        # This function is used to load part of the model parameters
        pretrain_model_weights = tf.saved_model.load(self.actor.checkpoint_dir)
        params_dict = {}
        for v in pretrain_model_weights.trainable_variables:
            params_dict[v.name] = v.read_value()
            # Load pre-training parameters to the model except for the last layer
        for idx, layer in enumerate(self.actor.variables[:-1]):
            layer.assign(pretrain_model_weights.variables[idx])

    def load_models(self):
        print(".... loading models ....")
        self.actor.load_weights(self.actor.checkpoint_file)

    def learn(self, episode, load_model):
        train = True

        with tf.GradientTape() as tape:
            # The reason for not taking Actor loss in DDPG is that we have to refine once and lose dependency
            # The logic for updating the actor loss here is the same as for the original network
            # Take the logits, calculate the actions, multiply the calculated actions by the logits in the network, and get the final loss for this set of actions
            batch_adj, batch_dict, batch_embedding, batch_net_input = self.environment.generate_batch(batch_size=self.batch_size)
            sparse_batch_net_input = sparse.vstack(batch_net_input)
            indices = list(zip(*sparse_batch_net_input.nonzero()))
            sparse_batch_net_input = tf.SparseTensor(indices=indices, values=np.float32(sparse_batch_net_input.data),
                                                    dense_shape=sparse_batch_net_input.get_shape())
            sparse_batch_net_input = tf.sparse.reshape(sparse_batch_net_input, [-1, self.graph.get_grf_size(), self.graph.get_grf_input_size()])
            states = tf.sparse.to_dense(sparse_batch_net_input)
            # states = tf.convert_to_tensor(batch_net_input, dtype=tf.float32)
            new_policy_logits = self.actor(batch_adj, states)
            new_action, mask = self.environment.action(actor_logits=new_policy_logits, train=True)
            new_rewards = self.environment.rewards(new_action)
            # extra_rewards = self.environment.reward_(episode, new_action, batch_dict)
            total_rewards = new_rewards # + extra_rewards
            # init baseline:
            if self.baseline is None:
                self.baseline = np.mean(total_rewards)
            else:
                self.baseline = self.baseline * 0.99 + np.mean(total_rewards) * 0.01
            # Because the following two operations require different variable types, we set two actions
            new_actions_i = tf.convert_to_tensor(new_action, dtype=tf.int32)

            # What if you don't use logits after refine?
            # refine_policy_logits = new_policy_logits/self.temperature
            refine_policy_logits = self.C * tf.tanh(new_policy_logits/self.temperature)

            # print("The output after using tanh", refine_policy_logits)
            refine_policy_logits = tf.where(mask,
                                            tf.ones_like(new_policy_logits) * (-np.inf),
                                            refine_policy_logits)

            # Add the probability sum of log, and there is no problem after testing
            neg_log_prob = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=refine_policy_logits,
                                                                                        labels=new_actions_i),
                                         axis=1)
            advantage = new_rewards - self.baseline
            # Do a normalization of advantage, but it doesn't feel very good
            # advantage = tf.convert_to_tensor((advantage-np.mean(advantage))/(np.std(advantage)+1e-10),
            #                                  dtype=tf.float32)
            actor_loss = tf.reduce_mean(advantage * neg_log_prob)

        # print("Episode: ", episode)
        # print("Action: \n", new_action)
        # print("Origin logits: \n", new_policy_logits.numpy()[0:3])
        # print("Refine logits: \n", refine_policy_logits.numpy()[0:3])
        # print("reward: \n", np.mean(new_rewards))

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
        if (episode+1) % 500 == 0:
            with self.train_summary_writer.as_default():
                tf.summary.scalar("actor_loss", actor_loss, step=episode)
                tf.summary.scalar("baseline", self.baseline, step=episode)
                tf.summary.scalar("best_reward", np.max(new_rewards), step=episode)
                tf.summary.scalar("mean_reward", np.mean(new_rewards), step=episode)
                # tf.summary.scalar("neg_log_prob", tf.reduce_mean(neg_log_prob), step=episode)
            print("Episode: ", (episode+1), end=" ")
            print("II: ", self.ii)
            print("本轮的actions: \n", repr(new_action[0:10]))
            print(np.mean(total_rewards))
            print("----------------------------------")
        if np.max(total_rewards) == 0:
            index = np.argmax(total_rewards)
            correct = new_action[index]
            print("Success Mapping!")
            print("total episode ", episode)
            print("Result: \n", correct)
            self.environment.show_placer(correct, batch_embedding[index])
            return True
        else:
            return False

