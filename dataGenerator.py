from scipy import sparse
import numpy as np


class DataGenerator:
    def __init__(self, graph):
        self.adj_dict = None
        self.graph = graph
        self.graph_ini()

    def graph_ini(self):
        adj = self.graph.adj
        refine_matrix = []
        node_num = np.size(adj, 1)
        for i in range(node_num):
            row_array = []
            node_idx = 1
            row_array.append(i + 1)
            for j in adj[i]:
                if j == 1:
                    row_array.append(node_idx)
                    node_idx += 1
                if j == 0:
                    node_idx += 1
            refine_matrix.append(row_array)
        adj_dict = {}
        # Convert the adjacency list to a dictionary
        for i in refine_matrix:
            a = i[1:]
            b = []
            for n in a:
                if n != 0:
                    b.append(n)
            adj_dict[i[0]] = b
        self.adj_dict = adj_dict

    def normalize_adj(self, adj):
        # normalization
        adj = adj + np.identity(len(adj))
        rowsum = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        normalized_adj = np.dot(np.dot(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return normalized_adj

    # Depth-first search
    def DFS(self, s):
        adj_dict = self.adj_dict
        stack = []
        stack.append(s)
        seen = []
        seen.append(s)
        sort = []
        while stack:
            vertex = stack.pop()  # Stack, take the last one and delete first-in-last-out
            nodes = adj_dict[vertex]
            for w in nodes:
                if w not in seen:
                    stack.append(w)
                    seen.append(w)
            sort.append(vertex)
        return sort

    # Breadth-first search
    def BFS(self, s):
        adj_dict = self.adj_dict
        queue = []
        queue.append(s)
        seen = []
        seen.append(s)
        sort = []
        while queue:
            vertex = queue.pop(0)  # Queue, first in, first out
            nodes = adj_dict[vertex]
            for w in nodes:
                if w not in seen:
                    queue.append(w)
                    seen.append(w)
            sort.append(vertex)
        return sort

    def sort2embedding(self, sort):
        embedding_new = self.graph.embedding.copy()
        adj_new = self.graph.adj.copy()  # + np.identity(len(self.graph.adj))
        net_input = self.graph.net_input.copy()
        source_adj_list = np.delete(embedding_new, 1, axis=1)
        adj_graph = np.delete(source_adj_list, -1, axis=1)
        # Input data processing
        adj_graph_new = []
        net_input_new = []
        dict = {}
        timestep = {}
        route = {}
        # Establish time department, routing, new and old node mapping
        t = embedding_new[:, 1]
        a = np.arange(1, len(sort) + 1, 1)
        r = embedding_new[:, -1]
        j = 0
        for i in sort:
            timestep[i] = t[i - 1]
            dict[i] = a[j]
            route[i] = r[i - 1]
            j += 1
        # Process sort, placing the routing node after the normal node
        list1 = list()
        for key in route.keys():
            list1.append(key)
        list1.sort()
        list2 = list()
        for i in list1:
            list2.append({i: route[i]})
        for d in list2:
            for key in d:
                if d[key] != 0:
                    sort.append(key)
                    sort.remove(key)
        # print("sort:",sort)
        # new net_input
        for i in sort:
            net = net_input[i - 1]
            net_input_new.append(net)
        net_input_new = np.array(net_input_new)
        # new adj_list
        for i in sort:
            node = adj_graph[i - 1]
            adj_graph_new.append(node)
        adj_graph_new = np.array(adj_graph_new)
        # new time array
        time_new = {}
        for i in sort:
            time_new[i] = t[i - 1]
        time_new = np.array(list(time_new.values()))
        # new array
        route_new = {}
        for i in sort:
            route_new[i] = r[i - 1]
        route_new = np.array(list(route_new.values()))
        embedding_new = np.insert(adj_graph_new, 1, time_new, axis=1)
        embedding_new = np.c_[embedding_new, route_new]
        # From the adjacency list to the adjacency matrix
        dict_new = {}
        j = 0
        for i in sort:
            dict_new[i] = j
            j += 1
        adj_row = []
        adj_last = []
        for key in dict_new.keys():
            adj_row.append(list(adj_new[key - 1]))
        adj_row = np.array(adj_row)
        for key in dict_new.keys():
            adj_last.append(list(adj_row[:, key - 1]))
        adj_last = np.array(adj_last)
        # Delete the last recomp column
        adj_graph_new = np.delete(adj_graph_new, -1, axis=1)
        return adj_last, adj_graph_new, embedding_new, net_input_new, dict

    def generate(self):
        # This function is used to generate all the breadth-first and depth-first adjacency matrices and the corresponding embedding information (including node number, time step information, adjacent nodes, and whether it is a routing node).
        # Returns information for all adjacency matrices and the corresponding embedding
        node_nums = self.graph.get_grf_size()
        feature_size = self.graph.get_grf_input_size()
        adj_matrix_list = []
        adj_list = []
        embedding_list = []
        dict_list = []
        net_input_list = []

        for starting_node in range(self.graph.get_grf_size()):
            # Each node does DFS and BFS separately as the starting point, and then saves it in the file
            # Since starting_node starts at 0, we need to do +1 once
            sort = self.BFS(starting_node+1)
            adj_matrix, adj, embedding, net_input, dict = self.sort2embedding(sort)
            adj_matrix_list.append(self.normalize_adj(adj_matrix))
            # adj_matrix_list.append(adj_matrix)
            adj_list.append(adj)
            embedding_list.append(embedding)
            dict_list.append(dict)
            # net_input_list.append(net_input)
            net_input_list.append(sparse.csr_matrix(np.reshape(net_input, [1, node_nums*feature_size])))
            sort = self.DFS(starting_node+1)
            adj_matrix, adj, embedding, net_input, dict = self.sort2embedding(sort)
            adj_matrix_list.append(self.normalize_adj(adj_matrix))
            # adj_matrix_list.append(adj_matrix)
            adj_list.append(adj)
            embedding_list.append(embedding)
            dict_list.append(dict)
            # net_input_list.append(net_input)
            net_input_list.append(sparse.csr_matrix(np.reshape(net_input, [1, node_nums*feature_size])))

        return np.array(adj_matrix_list, dtype=np.float), np.array(embedding_list), np.array(adj_list),  np.array(net_input_list), np.array(dict_list)

    def generate_original(self):
        adj_matrix_list = []
        adj_list = []
        embedding_list = []
        dict_list = []

        adj_matrix_list.append(self.normalize_adj(self.graph.adj))
        embedding_list.append(self.graph.embedding)
        adj_list.append(self.graph.graph)
        dict_list.append(self.adj_dict)

        return np.array(adj_matrix_list, dtype=np.float), np.array(embedding_list), np.array(adj_list), np.array(dict_list)

