import numpy as np
import hashlib

# TODO：
#  1. The last load node that can be executed is set up
#  2. When can I save? When can I go to load? Under what circumstances?

# This variable represents points that must not be mapped, such as normal nodes that must not be mapped to LRF, GRF, or memory
absolute_infeasible_prob = -np.inf
# The following variable represents the number of repeated mappings that can be tolerated
infeasible_prob = -100000


def abs_infeasible_update(batch_index, i, nodes, logits, mask):
    # This function adjusts unmappable nodes to negative infinity and changes the corresponding mask
    # batch_index : the batch number
    # i           : The logits of the node needs to be modified
    for node in nodes:
        logits[node] = absolute_infeasible_prob
        mask[batch_index][i][node] = 1


def infeasible_update(batch_index, i, nodes, logits, mask):
    # This function adjusts unmappable nodes to negative infinity and changes the corresponding mask
    # batch_index :the batch number
    # i           :The logits of the node needs to be modified
    for node in nodes:
        logits[node] = infeasible_prob
        mask[batch_index][i][node] = 1


class Environment:
    def __init__(self, action_dims, memory_mode, max_memory, max_GRF, max_LRF, C, temperature, ii, pea_width, total_adj,
                 total_embedding, total_graph, total_dict, total_net_input, reward_mode, beta=0.2):
        self.batch_index = None
        self.max_LRF = max_LRF
        self.max_GRF = max_GRF
        self.max_memory = max_memory
        # Whether to start memory. True indicates that memory is started and False indicates that memory is not started
        self.memory_mode = memory_mode
        self.C = C
        self.temperature = temperature
        self.beta = beta
        self.ii = ii
        self.pea_width = pea_width
        self.pea_size = pea_width * pea_width
        self.action_dims = action_dims
        self.total_net_input = total_net_input
        self.total_adj = total_adj
        self.total_embedding = total_embedding
        self.total_graph = total_graph
        self.total_dict = total_dict
        self.mapping_table = dict()
        self.reward_mode = reward_mode
        self.degree_matrix = self.degree(total_graph[0])

    def action(self, actor_logits, train):
        # This function is used to give feedback on what action to take when logits is available
        pea_size = self.pea_size
        # Define the LRF position to be pea_size and the GRF position to be pea_size+1
        LRF_pos = pea_size
        GRF_pos = pea_size + 1
        save_pos = pea_size + 2
        remain_pos= pea_size + 3
        load_pos = pea_size + 4

        # For normal nodes, LRF, GRF, and memory cannot perform mapping
        if self.memory_mode:
            # If memory is enabled, the entire probability of the memory portion needs to be modified
            normal_infeasible = [LRF_pos, GRF_pos, save_pos, load_pos, remain_pos]
        else:
            normal_infeasible = [LRF_pos, GRF_pos]
        # print(src_node_num)
        # This variable is used to read in the time step for each node
        actor_logits = actor_logits.numpy().copy()
        node_IDs = self.total_embedding[self.batch_index, :, 0]
        time_layer = self.total_embedding[self.batch_index, :, 1]
        # This variable is used to read in whether each node is a routing node, and if it is a routing node, then its value represents its passed value
        routing = self.total_embedding[self.batch_index, :, -1]
        # This variable is used to pass whether each node is a recomp node
        recomputing = self.total_embedding[self.batch_index, :, -2]
        batch_size, node_nums, _ = actor_logits.shape
        # print(node_nums)
        predicted_ids = np.ones((batch_size, node_nums), dtype=np.int)
        # One more location for LRF and one more location for GRF
        mask = np.zeros(shape=[batch_size, node_nums, self.action_dims])
        for batch in range(batch_size):
            cur_graph = self.total_graph[self.batch_index][batch]
            cur_embedding = self.total_embedding[self.batch_index][batch]
            new_predicted_ids = []
            # The following list is almost identical to what new_predicted_ids stores, but it will store exactly which PEA the LRF is stored in
            true_position = []
            infeasible = [[] for _ in range(self.ii)]
            # The keys of this dictionary are made up of two things, time step and position
            LRF_dict = {}
            LRF_ts_dict = {}
            GRF_dict = {}
            GRF_ts_dict = {}
            memory_dict = {}
            LD_index_dict = {}
            last_node_dict = {}
            recomp_dict = {}
            # Set a save_flag to record whether a save to memory operation has occurred
            # By default, no save or load operation is generated at the beginning
            save_flag = False
            load_flag = False
            for i in range(node_nums):
                # The condition is too strong, now it is weakened to only non-repetition
                # set the weight on all infeasible actions to -inf
                cur_node_id = node_IDs[batch, i]
                # What floor is this point on first
                cur_time_step = time_layer[batch, i]
                cur_time_layer = time_layer[batch, i] % self.ii
                # Take a look at the information stored by the current routing node
                cur_routing = routing[batch, i]
                cur_recomputing = recomputing[batch, i]
                # print("----------Time Layer: %d---------" % cur_time_layer)
                # print("----------Routing node information: %d---------" % cur_routing)
                cur_infeasible = infeasible[cur_time_layer].copy()
                # For nodes of recomp, open duplicate mappings
                if cur_recomputing != 0:
                    recomp_indexes = np.argwhere(cur_embedding[:, -2] == cur_recomputing)
                    if i == recomp_indexes[0]:
                        # This indicates the first recomp node
                        # Store his infeasible location
                        recomp_dict[cur_recomputing] = len(cur_infeasible)
                    else:
                        # This indicates a second recomp node
                        # Read the infeasible position of the previous recomp
                        temp_recomp_pos = recomp_dict[cur_recomputing]
                        cur_infeasible = np.delete(cur_infeasible, temp_recomp_pos)
                new_logits = np.array(actor_logits[batch][i])/self.temperature
                # Here you can set the tanh layer
                new_logits = self.C * np.tanh(new_logits)
                # print(new_logits)
                # TODO If cur_infeasible is not feasible enough to fill all nodes, then an update operation is performed, or if all nodes are already occupied, then a random mapping is performed
                if len(cur_infeasible) < self.pea_size:
                    # print(cur_infeasible)
                    # print("==========================================================")
                    infeasible_update(batch, i, nodes=cur_infeasible, logits=new_logits, mask=mask)
                # Determine whether it is a routing node. If it is not a routing node, it cannot be mapped to the register
                if cur_routing == 0:
                    abs_infeasible_update(batch, i, nodes=normal_infeasible, logits=new_logits, mask=mask)
                else:
                    dominate_point_indexes = np.argwhere(cur_graph == cur_node_id)
                    for index in dominate_point_indexes:
                        if index[1] != 0:
                            occup_LRF_flag = True
                            occup_GRF_flag = True
                            dominate_point_index = index[0]
                            # Find out where its parent really maps
                            true_dominate_mapping_point = true_position[dominate_point_index]
                            # Find out if the parent is mapped to LRF, GRF, or memory
                            dominate_mapping_point = new_predicted_ids[dominate_point_index]
                            # You need to check whether a particular location on a particular time step is occupied to determine whether it can be mapped to the LRF
                            LRF_ts_pos = str(cur_time_layer) + "," + str(true_dominate_mapping_point)
                            # You need to check whether the GRF on a specific time step is occupied to determine whether it can be mapped to the GRF
                            GRF_ts = str(cur_time_layer)
                            # View the mapped values in this location
                            occupation_LRF_ts = LRF_ts_dict.get(LRF_ts_pos, [])
                            occupation_LRF = LRF_dict.get(LRF_ts_pos, [])
                            occupation_GRF_ts = GRF_ts_dict.get(GRF_ts, [])
                            occupation_GRF = GRF_dict.get(GRF_ts, [])

                            if (len(occupation_LRF_ts) != 0) and ((cur_time_step not in occupation_LRF_ts) or
                                                                  (cur_routing not in occupation_LRF)):
                                # If the LRF is not empty and the time step of the occupied LRF is inconsistent with the current one or the stored value of the LRF is inconsistent with the current one
                                occup_LRF_flag = False

                            if (len(occupation_GRF_ts) != 0) and ((cur_time_step not in occupation_GRF_ts) or
                                                                  (cur_routing not in occupation_GRF)):
                                # If the GRF is not empty and the time step of the occupied GRF is inconsistent with the current one or the stored value of the GRF is inconsistent with the current one
                                occup_GRF_flag = False

                            # First check whether it can be mapped to the LRF, because there may be a route node corresponding to the PEA location that has already been mapped
                            # If the parent is mapped to the LRF, then it cannot be mapped to the GRF
                            if (not occup_LRF_flag) or dominate_mapping_point == GRF_pos:
                                new_logits[LRF_pos] = absolute_infeasible_prob
                                mask[batch][i][LRF_pos] = 1
                            # Then check to see if it maps to the GRF
                            # If the parent is mapped to the LRF, then it cannot be mapped to the GRF
                            if (not occup_GRF_flag) or dominate_mapping_point == LRF_pos:
                                new_logits[GRF_pos] = absolute_infeasible_prob
                                mask[batch][i][GRF_pos] = 1

                            if self.memory_mode:
                                LD_indexes = LD_index_dict.get(cur_routing, [])
                                if len(LD_indexes) == 0:
                                    # In other words, if you can't find the dependency in this dict, start looking for it
                                    LD_indexes = np.argwhere(cur_embedding[:, -1] == cur_routing)
                                    LD_index_dict[cur_routing] = LD_indexes
                                    # Update save_flag and load_flag
                                    save_flag = False
                                    load_flag = False
                                    # Only when the number of long-dependent nodes is long enough will the judgment need to be performed
                                    # First define the last node that can be executed as None. When determining whether it can be mapped, first determine whether this node is None
                                    last_node = None
                                    # Iterate over all the nodes of the long dependency to find the last node where the load can be executed
                                    for LD_index in LD_indexes:
                                        temp_cur_time = cur_embedding[LD_index, 1]
                                        temp_memory_pos = str(temp_cur_time)
                                        temp_memory_occupation = memory_dict.get(temp_memory_pos, [])
                                        if len(temp_memory_occupation) < self.max_memory:
                                            last_node = LD_index
                                    last_node_dict[cur_routing] = last_node

                                # After dealing with the long dependency for the first time, it's time to formally deal with the execution
                                last_node = last_node_dict[cur_routing]
                                # The length of the long dependence must be greater than or equal to 3
                                if len(LD_indexes) >= 3:
                                    memory_ts = str(cur_time_layer)
                                    memory_occupation = memory_dict.get(memory_ts, [])
                                    # If the save operation has not been performed
                                    if not save_flag:
                                        # If any of the following conditions are met, it cannot be mapped to save_pos
                                        # 1. Get the last_node. If the last_node is empty, there is no place to load on the long dependency, so the save operation cannot be performed
                                        # 2. If the last node that can perform the load position is too close to the current node, the save, load, and remain operations cannot be performed either
                                        # The value of cur_node_id starts from 1 and the value of last_node starts from 0
                                        # 3. The parent node is a local register or a global register
                                        # 4. The save_load resource in the time step of the current node is fully occupied, and the long dependency information of the node that occupies the resource is inconsistent with the current one
                                        if (not last_node) or ((last_node-(cur_node_id-1)) < 2) or \
                                                (dominate_mapping_point in [LRF_pos, GRF_pos]) or \
                                                (len(memory_occupation) >= self.max_memory):
                                            abs_infeasible_update(batch, i, nodes=[save_pos, remain_pos, load_pos],
                                                              logits=new_logits, mask=mask)
                                        else:
                                            # The node can then be mapped to save_pos
                                            abs_infeasible_update(batch, i, nodes=[remain_pos, load_pos],
                                                                  logits=new_logits, mask=mask)
                                            # new_logits[save_pos] = 10000000
                                    elif dominate_mapping_point == load_pos:
                                        # If the previous node has performed the load operation, then the next node can only return to the normal node:
                                        abs_infeasible_update(batch, i, nodes=normal_infeasible, logits=new_logits,
                                                          mask=mask)
                                    elif load_flag:
                                        # When load_flag takes effect, the load operation has been performed, and the remaining long dependencies cannot be mapped to memory
                                        abs_infeasible_update(batch, i, nodes=[save_pos, remain_pos, load_pos],
                                                          logits=new_logits, mask=mask)
                                    else:
                                        # If the node has reached the last_node, it must perform the load operation
                                        if cur_node_id-1 == last_node:
                                            # Mappings from 0 to pea_size+3 cannot be performed, and only load_pos can be mapped properly
                                            abs_infeasible_update(batch, i, nodes=range(pea_size+4), logits=new_logits,
                                                              mask=mask)
                                        else:
                                            # If the save operation has been performed and the last_node has not been accessed, it now has only two places to select from
                                            # Mapping from 0 to pea_size+2 cannot be performed, only remain_pos and load_pos can be mapped properly
                                            abs_infeasible_update(batch, i, nodes=range(pea_size+3), logits=new_logits,
                                                                  mask=mask)
                                            # Only when the previous node is a remain node can the load operation be executed. In this way, the interval between save and load is guaranteed to be at least two time steps
                                            if dominate_mapping_point != remain_pos:
                                                abs_infeasible_update(batch, i, nodes=[load_pos], logits=new_logits,
                                                                  mask=mask)
                                else:
                                    # The length of this group dependence is less than 3, and it passes directly
                                    abs_infeasible_update(batch, i, nodes=[save_pos, remain_pos, load_pos],
                                                      logits=new_logits, mask=mask)

                new_logits = (new_logits - np.max(new_logits))
                probs = np.exp(new_logits) / np.sum(np.exp(new_logits))
                # print(probs)
                # In training and in testing
                # greedy is used if testing, or stochastic is used if training
                if train:
                    action = np.random.choice(self.action_dims, 1, p=probs)
                    new_predicted_ids.append(action[0])
                    # If it does map to the LRF, update the dictionary of time steps and location information
                    if action[0] == LRF_pos:
                        true_position.append(true_position[dominate_point_index])
                        if cur_routing not in occupation_LRF:
                            occupation_LRF.append(cur_routing)
                        if cur_time_step not in occupation_LRF_ts:
                            occupation_LRF_ts.append(cur_time_step)
                        LRF_dict[LRF_ts_pos] = occupation_LRF
                        LRF_ts_dict[LRF_ts_pos] = occupation_LRF_ts

                    # If it does map to GRF
                    elif action[0] == GRF_pos:
                        true_position.append(action[0])
                        if cur_routing not in occupation_GRF:
                            occupation_GRF.append(cur_routing)
                        if cur_time_step not in occupation_GRF_ts:
                            occupation_GRF_ts.append(cur_time_step)
                        GRF_dict[GRF_ts] = occupation_GRF
                        GRF_ts_dict[GRF_ts] = occupation_GRF_ts

                    elif action[0] == save_pos:
                        true_position.append(action[0])
                        # If it is mapped to the save location, adjust save_flag
                        save_flag = True
                        memory_occupation.append(cur_routing)
                        memory_dict[memory_ts] = memory_occupation

                    elif action[0] == remain_pos:
                        true_position.append(action[0])

                    elif action[0] == load_pos:
                        true_position.append(action[0])
                        # If it is mapped to the load location, adjust load_flag
                        load_flag = True
                        # load takes up extra resources, so save the node in the load position directly into the dict
                        memory_occupation.append(cur_routing)
                        memory_dict[memory_ts] = memory_occupation
                    else:
                        true_position.append(action[0])
                        infeasible[cur_time_layer].append((action[0]))
                    # print("Mapped node: ", action[0])
                    # print("The current occupied location: ", true_position)
                    # print(true_position)
                else:
                    # argmax is straight out of scalar, so no indexing operations like action[0] are required
                    action = np.argmax(probs)
                    new_predicted_ids.append(action)
                    # If mapped to a register, additional design logic is required to limit how many nodes can be mapped to the LRF or GRF
                    if action == LRF_pos or action == GRF_pos:
                        continue
                    else:
                        infeasible[cur_time_layer].append(action)
                # print(new_predicted_ids)
            predicted_ids[batch] = new_predicted_ids
            # print(mask)
        return predicted_ids, mask

    def get_distance_between(self, target_node, neighbour_target_node):
        #This function is used to construct the Manhattan distance of the normal and torus structures
        dim_n = self.pea_width
        mode = self.reward_mode
        if mode == 1:
            # normal structure of Manhattan distance
            node_xy = np.array(divmod(target_node, dim_n))
            neighbour_xy = np.array(divmod(neighbour_target_node, dim_n))
            distance = np.sum(np.abs(np.subtract(node_xy, neighbour_xy)))
            return distance
        elif mode == 2:
            # torus structure distance from Manhattan
            node_xy = np.array(divmod(target_node, dim_n))
            # Replace the parent node with the column and column coordinates on CGRA, starting with 0
            neighbour_xy = np.array(divmod(neighbour_target_node, dim_n))
            """
            # Replace the child node with the column and column coordinates on the CGRA, starting with 0
            dis_x = neighbour_xy[0] - node_xy[0]
            # print("dis_x", dis_x)
            dis_y = neighbour_xy[1] - node_xy[1]
            # print("dis_y", dis_y)
            if dis_x > (dim_n - dis_x):
                # print("-dis_x", dim_n - dis_x)
                node_xy[0] = dim_n + node_xy[0]
                # print("node_xy[0]", node_xy[0])
            if dis_y > (dim_n - dis_y):
                # print("-dis_y", dim_n - dis_y)
                node_xy[1] = dim_n + node_xy[1]
                # print("node_xy[1]", node_xy[1])
            distance = np.sum(np.abs(np.subtract(node_xy, neighbour_xy)))
            """
            dis_x = abs(neighbour_xy[0] - node_xy[0])
            dis_y = abs(neighbour_xy[1] - node_xy[1])
            dis_x = min(dis_x,self.pea_width-dis_x)
            dis_y = min(dis_y,self.pea_width-dis_y)          
            distance = dis_x+dis_y
            return distance
        elif mode == 3:
            # Diagonal structure of Manhattan distance
            node_xy = np.array(divmod(target_node, dim_n))
            neighbour_xy = np.array(divmod(neighbour_target_node, dim_n))
            distance = np.max(np.abs(np.subtract(node_xy, neighbour_xy)))            
           
            return distance
        elif mode == 4:
            # 1-hop structure of Manhattan distance
            node_xy = np.array(divmod(target_node, dim_n))
            neighbour_xy = np.array(divmod(neighbour_target_node, dim_n))
            distance = np.sum((np.abs(np.subtract(node_xy, neighbour_xy))+1)//2)            
           
            return distance


    def repeat_timecheck(self, timestep, actions):
        # Check whether the time section is repeated: 1. The first one in the current time section is not counted, and the second one is punished
        ii = self.ii
        # print("ii",ii)
        t_check = np.zeros(len(actions))
        action_new = np.array(actions)
        timeslot = map(lambda x: x % ii, timestep)  # Find the timeslot list
        ts = list(timeslot)
        # print("ts",ts)
        action_set = set(actions)
        actions = actions.tolist()
        # print("action:",actions)
        # print("action_s",action_set)
        for val in action_set:
            if actions.count(val) != 1:  # When the action is repeated
                w = np.argwhere(action_new == val)  # Find all nodes whose action is the same as the current location
                w = w.flatten()
                #print(w)
                for i in w:
                    for j in w:
                        if ts[j] == ts[i] and i != j and i != w[0]:
                            t_check[i] = 1
        return t_check

    def repeat_recheck(self, recomputation, actions):  # Check whether recomputaion is activated when the node is repeated
        re_check = np.ones(len(actions))  # Create an empty list
        action_new = np.array(actions)
        re = recomputation
        actions = actions.tolist()
        for i, val in enumerate(actions):
            if actions.count(val) != 1:  # When the action is repeated
                w = np.argwhere(action_new == val)  # Find all nodes whose action is the same as the current location
                w = w.flatten()
                for j in w:
                    if re[j] == re[i] and i != j and re[i] != 0:
                        re_check[i] = 0
        return (re_check)

    def repeat_maxcheck(self,index,actions):
        G_max = self.max_GRF
        pea_size = self.pea_size
        max_check = np.zeros(len(actions))
        action_new = np.array(actions)
        action_set = set(actions)
        actions = actions.tolist()
        for val in action_set:
            n = actions.count(val)
            if n != 1:  # When the action is repeated
                w = np.argwhere(action_new == val)  # Find all nodes whose action is the same as the current location
                w = w.flatten()
                if val == pea_size:  #LRF Repetitive time
                    action_real = self.action2mapping(action_new,self.total_embedding[self.batch_index[index]])
                    repeat_real=[]
                    for i,act in enumerate(action_new):
                        if action_new [i] == pea_size:
                            repeat_real.append(action_real[i])
                        s_repeat_real = set(repeat_real)
                        if len(s_repeat_real) != len(repeat_real):
                            max_check[i] = 1
                if val == pea_size + 1 and n > G_max:  # The number of special nodes is greater than the maximum value
                    for j in w:
                        if j != w[0]:
                            max_check[j] = 1  # The second node starts adding penalties
       # print(max_check)
        return max_check


    def punish(self, index, timestep, actions, recomputation):
        pea_size = self.pea_size
        t_list = self.repeat_timecheck(timestep, actions)
       # print("t_check",t_list)
        re_list = self.repeat_recheck(recomputation, actions)
       # print("re_check",re_list)
        max_list = self.repeat_maxcheck(index,actions)
       # print("max_check",max_list)
        punish = np.zeros(len(actions))
        for i, val in enumerate(actions):  # Normal pe mapping
            if val < pea_size:
                punish[i] = t_list[i] and re_list[i]
            else:  # When the policy node is reached (LRF,GRF, etc.)
                punish[i] = t_list[i] and re_list[i] and max_list[i]
       # print("punish:",punish)
       # print("--------------------------------------------------------")
        return punish

    def reward_(self, episode, actions, dicts):
        dict_new = {}
        mapping = {}
        actions_new = []
        for i in range(len(actions)):
            dict = dicts[i]
            for k, v in dict.items():
                dict_new[v] = k
            index = 1
            for map in actions[i]:
                key = dict_new[index]
                mapping[key] = map
                index += 1
            maplist = list(mapping.items())
            maplist.sort()
            action_new = []
            for j in maplist:
                action_new.append(j[-1])
            actions_new.append(action_new)
        actions_new = np.array(actions_new)
        extra_reward = np.zeros(len(actions_new))
        for batch_index, batch in enumerate(actions_new):
            batch = batch.view(np.uint8)
            hash_index = hashlib.sha1(batch).hexdigest()
            num = self.mapping_table.get(hash_index, 0)
            if num == 0:
                self.mapping_table[hash_index] = 1
            else:
                self.mapping_table[hash_index] += 1
            # The following is an extra reward added to the episode, in order to give him points if he encounters new situations in the later stages of the model
            # extra_reward[batch_index] = 2 * self.beta * (np.log(episode+1)) ** 0.5 * (num + 1) ** (-0.5)
            extra_reward[batch_index] = self.beta * (num + 1) ** (-0.5)
        return extra_reward

    def degree(self, source_adj_list):
        source_adj_list = source_adj_list.copy()
        degree_dict = {}
        for node in source_adj_list:
            node_degree = 0
            node_idx = int(node[0])
            # print("node_id:", node_idx)
            for i in node[1:]:
                if i == 0:
                    continue
                else:
                    node_degree += 1
            wnode = np.argwhere(source_adj_list == node_idx)
            node_degree += len(wnode) - 1
            degree_dict[node_idx] = node_degree
        degree_matrix = np.zeros([len(source_adj_list), len(source_adj_list)])

        min_degree = 10000
        for node in source_adj_list:
            node_idx = node[0]
            for neighbour_idx in node[1:]:
                if neighbour_idx == 0:
                    continue
                ngb_idx = int(neighbour_idx)
                total_degree = degree_dict[node_idx] + degree_dict[ngb_idx]
                degree_matrix[node_idx-1][ngb_idx-1] = total_degree
                degree_matrix[ngb_idx-1][node_idx-1] = total_degree
                if total_degree < min_degree:
                    min_degree = total_degree
        # degree_matrix = np.where(degree_matrix==0, min_degree, degree_matrix)
        # normalized_degree_matrix = degree_matrix
        normalized_degree_matrix = (degree_matrix - np.min(degree_matrix))/(np.max(degree_matrix)-np.min(degree_matrix))
        # print(normalized_degree_matrix)
        return normalized_degree_matrix

    def reward(self, index, action, punish):
        action_new = action.copy()
        LRF_positon = self.pea_size
        GRF_position = self.pea_size + 1
        save_position = self.pea_size + 2
        remain_position = self.pea_size + 3
        load_position = self.pea_size + 4
        source_adj_list = self.total_graph[self.batch_index[index]]
        degree_matrix = self.degree_matrix.copy()
        # print("c_dict",c_dict)
        # print(source_adj_list)
        # This function is used for feedback. What kind of reward will be given after the action is taken
        # Because it needs to be stored in buffer, it will no longer take the form of batch_size
        distance_sum = np.zeros(1, dtype=float)
        # Introduce additional reward calculations
        wrong_mapping = np.zeros(1, dtype=float)
        nodelist = source_adj_list[:, 0]
        # print("nodelist:", nodelist)
        # Create action and node mapping dictionary
        node_dict = {}
        action_dict = {}
        action_dict_unchange = {}
        for i in range(len(action_new)):
            action_dict_unchange[nodelist[i]] = action_new[i]
            action_dict[nodelist[i]] = action_new[i]
            node_dict[nodelist[i]] = i
        # print("Node dict", node_dict)
        # print("action_dict:", action_dict)
        # print("node_dict", node_dict)
        # Because our graph is fixed, there is no need to maintain a completely consistent adjacency list in the batch
        for id, node in enumerate(source_adj_list):
            # print(index, node)
            # This variable is used to record whether the parent node is in the LRF
            punish_flag = False
            LRF_flag = False
            GRF_flag = False
            save_flag = False
            remain_flag = False
            load_flag = False
            # node_idx:Node sequence number
            node_idx = int(node[0])
            # Find the corresponding location of the node in the original action
            target_node = action_dict_unchange[node_idx]  # target_node：Mapped location
            if target_node == LRF_positon:
                w = np.argwhere(source_adj_list == node_idx)
                for i in w:
                    if i[-1] != 0:
                        father_nd = nodelist[i[0]]
                        target_node = action_new[node_dict[father_nd]]
                        action_new[id] = action_new[node_dict[father_nd]]
                LRF_flag = True
            if target_node == GRF_position:  # The parent node is a GRF
                GRF_flag = True
            if target_node == save_position:  # The parent node is save
                save_flag = True
            if target_node == remain_position:  # The parent node is remain
                remain_flag = True
            if target_node == load_position:  # The parent node is load
                load_flag = True
            if punish[id] == 1:
                punish_flag = True
            # print("The node mapped when the parent node comes out of the network: ", action[node_idx-1])
            # print("Nodes obtained after refine: ", target_node)
            # Find child node
            for neighbour_idx in node[1:]:
                neighbour_punish_flag = False
                neighbour_LRF_flag = False
                neighbour_GRF_flag = False
                neighbour_save_flag = False
                neighbour_remain_flag = False
                neighbour_load_flag = False
                neighbour_idx = int(neighbour_idx)
                if neighbour_idx == 0:
                    continue
                neighbour_target_node = int(action_dict_unchange[neighbour_idx])
                if neighbour_target_node == LRF_positon:
                    neighbour_LRF_flag = True
                    w = np.argwhere(source_adj_list == neighbour_idx)
                    for i in w:
                        if i[-1] != 0:
                            father_nd = nodelist[i[0]]
                            neighbour_target_node = action_new[node_dict[father_nd]]
                            action_new[node_dict[neighbour_idx]] = action_new[node_dict[father_nd]]
                if neighbour_target_node == GRF_position:
                    neighbour_GRF_flag = True
                if neighbour_target_node == save_position:
                    neighbour_save_flag = True
                if neighbour_target_node == remain_position:
                    neighbour_remain_flag = True
                if neighbour_target_node == load_position:
                    neighbour_load_flag = True
                if punish[node_dict[neighbour_idx]] == 1:
                    neighbour_punish_flag = True
                # print("neighbourNode: %d" % neighbour_idx)
                # print("The node mapped when the child node comes out of the network: ", action[neighbour_idx-1])
                # print("Nodes obtained after refine: ", neighbour_target_node)
                d = degree_matrix[node_idx-1][neighbour_idx-1]
                # d = 1
                # print("Target node:", node_idx)
                # print("Neighbour node:", neighbour_idx)
                distance = self.get_distance_between(target_node, neighbour_target_node)
                # print("Distance", distance)
                if LRF_flag and (not neighbour_LRF_flag):
                    # In this case, if the parent node is mapped to the LRF and the child node is removed from the LRF, then, unless the distance is 0, all other mappings are mismapped and additional distance needs to be added
                    # print("The parent node is: ", node_idx, "Child node is: ", neighbour_idx)
                    # print("Here We Find one, distance is ", distance)
                    distance_sum += distance*d
                    wrong_mapping += 1
                elif GRF_flag or neighbour_GRF_flag or save_flag or neighbour_save_flag or remain_flag or \
                        neighbour_remain_flag or load_flag or neighbour_load_flag:
                    # If there is a GRF node, the distance between the GRF node and other nodes is 0.
                    distance = 0
                    distance_sum += distance
                else:
                    # The rest of the case is the same as the original, adding distance only when the distance is greater than 1
                    if distance > 1:
                        wrong_mapping += 1
                        distance_sum += distance*d

                if punish_flag or neighbour_punish_flag:
                    # print("punish active")
                    distance_sum = distance_sum + 10

        reward = - ((distance_sum + wrong_mapping*4) ** (1 / 3))
        return reward

    def rewards(self, actions):
        # This function is used to record the output of the actions of the entire batch
        batch_size, node_nums = actions.shape
        rewards = np.zeros(batch_size, dtype=float)
        for index, action in enumerate(actions):
            timestep = self.total_embedding[self.batch_index,:, 1][index]
            # print("timestep",timestep)
            recomputation = self.total_embedding[self.batch_index,:, -2][index]
            # ("recomputation",recomputation)
            punish = self.punish(index, timestep, action, recomputation)
            # print(punish)
            rewards[index] = self.reward(index=index, action=action, punish=punish)
        return rewards

    def action2mapping(self, action, graph):
        # This function is used to determine the position of the LRF, because in our model the LRF is at the position of pea_size and does not know its true position correctly
        # print(action)
        action_new = action.copy()
        LRF_positon = self.pea_size
        source_adj_list = graph
        for mapping_index, node in enumerate(source_adj_list):
            node_idx = int(node[0])  # node_idx:Node sequence number
            target_node = action_new[mapping_index]  # target_node：Mapped location
            if target_node == LRF_positon:
                w = np.argwhere(source_adj_list == node_idx)
                for i in w:
                    if i[-1] != 0:
                        action_new[mapping_index] = action_new[i[0]]
        return action_new

    def mapping2showing(self, actions, embedding):
        # This function is used to arrange the mapped nodes in output order
        # TODO:
        #  The Graph here needs to be adjusted manually if the length of the neighboring node changes
        graph = np.concatenate([np.expand_dims(embedding[:, 0], axis=1), embedding[:, 2:6]], axis=1)
        nodes_id = graph[:, 0]
        LRF_pos = self.pea_size
        mapping = self.action2mapping(actions, graph)
        saving_list = []
        ii = self.ii
        time_layer = embedding[:, 1] % ii

        for time_step in range(ii):
            saving_dict = {}
            ts_index = np.where(time_layer == time_step)[0]
            # Check the location of a round of normal node emissions
            for i in range(self.pea_size+5):
                cur_saving_list = []
                index = np.where(mapping[ts_index] == i)[0]
                for action_index, action in enumerate(actions[ts_index[index]]):
                    if action == LRF_pos:
                        cur_saving_list.append("({})".format(nodes_id[ts_index[index[action_index]]]))
                    else:
                        cur_saving_list.append(str(nodes_id[ts_index[index[action_index]]]))
                saving_dict[i] = cur_saving_list
            saving_list.append(saving_dict)
        return saving_list

    def show_placer(self, actions, embedding):
        mat1 = "{:^8}"
        mat2 = "{:^38}"
        mapping_list = self.mapping2showing(actions, embedding)

        for ii_layer, mapping_dict in enumerate(mapping_list):
            # Enter the first ii mapping
            print("-" * 16, "[", ii_layer, "]", "-" * 17)
            for i in range(self.pea_size):
                print("|", end='')
                mapping_nodes = mapping_dict.get(i)
                if i % self.pea_width == self.pea_width - 1:
                    # If it's 0 at the end of the module, that means it's the end of the line
                    print(mat1.format(" ".join(str(mapping_node) for mapping_node in mapping_nodes)), end='|\n')

                else:
                    print(mat1.format(" ".join(str(mapping_node) for mapping_node in mapping_nodes)), end='|')
            # This section is used to output GRF information
            print("-" * 17, "[GRF]", "-" * 16)
            mapping_nodes = mapping_dict.get(self.pea_size+1)
            print("|", end='')
            print(mat2.format(" ".join(str(mapping_node) for mapping_node in mapping_nodes)), end='|\n')
            # This section is used to output information from memory
            print("-" * 10, "[save|remain|load]", "-" * 10)
            print("|", end='')
            for i in [self.pea_size+2, self.pea_size+3, self.pea_size+4]:
                mapping_nodes = mapping_dict.get(i)
                print("{:^12}".format(" ".join(str(mapping_node) for mapping_node in mapping_nodes)), end="|")
            print("\n")

    def generate_batch(self, batch_size):
        # This function is used to generate a batch_size data set
        total_nums, _, _ = self.total_adj.shape
        # The batch_size data can be selected repeatedly by using the method of choice, which can also ensure the diversity of input data
        batch_index = np.random.choice(total_nums, batch_size, True)
        # Storing this batch_index in the class will change each time it is generated, and it will also facilitate subsequent generation of actions
        self.batch_index = batch_index
        return self.total_adj[batch_index], self.total_dict[batch_index], self.total_embedding[batch_index], \
               self.total_net_input[batch_index]





