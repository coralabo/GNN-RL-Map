import numpy as np


class Graph:
    def __init__(self, origin_embedding, ii):
        self.adj = None
        self.graph = None
        self.embedding = None
        self.net_input = None
        self.normalized_adj = None
        self.ii = ii
        self.read_in_graph(origin_embedding)
        self.init_adjacency_matrix()

    def read_in_graph(self, origin_embedding):
        def recomp(embedding, net_input, nums_ngb):
            recomp_embedding = embedding.copy()
            recomp_net_input = net_input.copy()
            # The purpose of this function is to do recomputation
            _, feature_nums = recomp_embedding.shape
            _, net_input_feature = net_input.shape
            # Find the location of all the neighbours
            ngb_pos = [(i + 1) * 2 for i in range(nums_ngb)]
            # check the neighbour. If the neighbour is non-zero, record the value as 1
            ngb_mask = np.where(recomp_embedding[:, ngb_pos] != 0, 1, 0)
            # Record the true number of neighbors for each node
            real_ngb_nums = np.sum(ngb_mask, axis=1)
            # Find all the nodes that need to perform recomp, that is, if the number of child nodes is greater than 2, you need to perform the split operation
            recomp_indexes = np.where(real_ngb_nums > 10)[0]

            for recomp_index in recomp_indexes:
                recomp_node = recomp_embedding[recomp_index]
                new_comp_node = np.zeros(feature_nums, dtype=int)
                new_input_comp_node = np.zeros(net_input_feature, dtype=int)
                # The first part is the number of the original node, and the second part is the time step of the original node
                new_input_comp_node[recomp_node[0]-1] = 1
                new_input_comp_node[len(embedding)+recomp_node[1]] = 1
                # Change the second-to-last column to 1, which indicates the reomp node
                new_input_comp_node[-2] = 1
                # Change the maximum number of nodes in the first column plus one
                new_comp_node[0] = recomp_embedding[-1][0] + 1
                # The time step is the same as the original time step
                new_comp_node[1] = recomp_node[1]
                # Since the maximum number of children is 4, the last two children are directly assigned to the newly generated point
                new_comp_node[2:6] = recomp_node[6:10]
                # Modify the adjacency of the original node
                recomp_embedding[recomp_index][6:10] = 0
                # Example Modify the relationship between the newly added node and the original node
                # Find the parent of the recomp node in the original data
                farther_indexes = np.argwhere(recomp_embedding[:, [2, 4, 6]] == recomp_node[0])
                for farther_index in farther_indexes[:, 0]:
                    # Look for the first open spot
                    vacant_pos = np.argwhere(recomp_embedding[farther_index][[4, 6, 8]] == 0)[0][0]
                    # Select the first vacant position as the filling position and change it to the number of the newly added node
                    recomp_embedding[farther_index][(vacant_pos + 2) * 2] = recomp_embedding[-1][0] + 1
                # Change the number of the original node in the second-to-last column
                recomp_node[-2] = recomp_node[0]
                new_comp_node[-2] = recomp_node[0]
                # Put the finished recomp nodes into the raw data
                recomp_embedding = np.append(recomp_embedding, [new_comp_node], axis=0)
                recomp_net_input = np.append(recomp_net_input, [new_input_comp_node], axis=0)

            return recomp_embedding, recomp_net_input

        # graph just stores the adjacency matrix
        # The embedding includes the adjacency matrix, time step and whether it is a routing node
        embedding = origin_embedding.copy()
        # Add an additional network input
        # The first part of net_input uses one-hot to represent the node number of each node (ignoring recomp nodes and later added routing nodes).
        node_number = np.identity(len(embedding))
        # The second part of net_input uses one-hot to represent the time step for each node
        node_timestep = np.zeros([len(embedding), max(embedding[:, 1]) + 1], dtype=np.int)
        node_timestep[range(len(embedding)), embedding[range(len(embedding)), 1]] = 1
        # The last two parts of net_input represent whether they are recomp nodes and whether they are routing nodes, respectively
        node_recomp_routing = np.zeros([len(embedding), 2], dtype=np.int)
        net_input = np.concatenate([node_number, node_timestep, node_recomp_routing], axis=1)
        _, net_feature_num = np.shape(net_input)
        # TODO: This parameter is used to store how many child nodes there are, and will be called as a parameter if possible
        nums_ngb = 4
        # Procedure for executing recomp
        embedding, net_input = recomp(embedding, net_input, nums_ngb)
        _, feature_num = np.shape(embedding)
        embedding = np.array(embedding)

        # Long-dependent nodes are turned into routing nodes and stored in the embedding
        for nodes in embedding:
            cur_node = nodes[0]
            # print("The current node is: ", cur_node)
            cur_node_ts = nodes[1]
            for j in range(nums_ngb):
                neighbour_node = nodes[2 * (j + 1)]
                neighbour_index = 2 * (j + 1)
                edge_type = nodes[2 * (j + 1) + 1]
                if neighbour_node == 0:
                    break
                # print("The current neighbor node is: ", neighbour_node)
                neighbour_node_ts = embedding[neighbour_node - 1][1]
                time_difference = neighbour_node_ts - cur_node_ts
                if (time_difference > 1) or (edge_type != 0):
                    # print("We found one with a big time step gap")
                    nums_routing = time_difference - 1 + 2 * edge_type
                    insert_array = np.zeros([nums_routing, feature_num], dtype=int)
                    net_insert_array = np.zeros([nums_routing, net_feature_num], dtype=int)
                    # Modify the information they store in the last column to represent the data they transmit
                    insert_array[:, -1] = cur_node
                    # Change the last column to represent the routing node, and change the first part to represent the information passed by routing
                    net_insert_array[:, [cur_node - 1, -1]] = 1
                    cur_max_node = embedding[-1][0]
                    for i in range(nums_routing):
                        # Register the information about the inserted node and the time step
                        insert_array[i][0] = cur_max_node + i + 1
                        insert_array[i][1] = cur_node_ts + i + 1
                        net_insert_array[i][len(origin_embedding)+cur_node_ts + i + 1] = 1
                        # Special handling is required for the beginning and ending nodes
                        # The beginning node needs to process the connection relationship between the parent node
                        if i == 0:
                            # Modify the association relationship of the original long dependent parent node
                            # Change the value to connect to the first routing node
                            embedding[cur_node - 1][neighbour_index] = insert_array[0][0]
                            # Change the original edge type to 0
                            embedding[cur_node - 1][neighbour_index + 1] = 0
                        # If it is a tail node, then it is connected to a long-dependent child node
                        if i == nums_routing - 1:
                            insert_array[i][2] = neighbour_node
                        # If it is not the tail node, then it is connected to the next routing node
                        else:
                            insert_array[i][2] = insert_array[i][0] + 1
                        # print(raw)
                    # print(insert_array)
                    embedding = np.concatenate([embedding, insert_array], axis=0)
                    net_input = np.concatenate([net_input, net_insert_array], axis=0)

        # TODO: wanting cols here also changes when the number of neighbors changes because the number of neighboring nodes in the middle changes
        wanting_cols = np.array([0, 1, 2, 4, 6, 8, -2, -1])
        embedding = embedding[:, wanting_cols]
        # Remove the time step and the column used to determine the routing node and recomp
        graph = np.delete(np.array(embedding), [1, -2, -1], axis=1)
        # print(graph)
        embedding = np.array(embedding)


        self.net_input = net_input
        self.graph = graph
        self.embedding = embedding

    def init_adjacency_matrix(self):
        graph = self.graph
        node_num = len(graph)
        adj = np.zeros([node_num, node_num], dtype=int)
        for i in range(node_num):
            for j in range(node_num):
                if i == j:
                    adj[i][j] = 0
                elif j + 1 in graph[i] or i + 1 in graph[j]:
                    adj[i][j] = 1
        self.adj = adj

        # normalization
        rowsum = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        normalized_adj = np.dot(np.dot(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        self.normalized_adj = normalized_adj

    def get_grf_size(self):
        # This function is used to give feedback on how many nodes there are
        return len(self.graph)

    def get_grf_input_size(self):
        # This function is used to give back feature_size for each node
        _, feature_size = self.net_input.shape
        return feature_size
