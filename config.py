import argparse

parser = argparse.ArgumentParser(description='Configuration file')

net_arg = parser.add_argument_group('Network')
net_arg.add_argument('--batch_size', type=int, default=64, help='batch size')
net_arg.add_argument('--pea_width', type=int, default=4, help='width of pea')
net_arg.add_argument('--src_file_path', type=str, default="data/input.txt",
                     help='file path of the source graph')
net_arg.add_argument('--hidden_dims', type=int, default=64, help='Dimensions in hidden layers')
net_arg.add_argument('--head_nums', type=int, default=4, help='Nums of head')
net_arg.add_argument('--actor_lr', type=float, default=1e-4, help='learning rate in network')
net_arg.add_argument('--max_iteration', type=int, default=1000, help='max iteration')
net_arg.add_argument('--ckpt_dir', type=str, default='tmp', help="path to save models")
net_arg.add_argument('--temperature', type=float, default=10, help="temperature of softmax")
net_arg.add_argument('--beta', type=float, default=0.2, help="parameter of extra reward")
net_arg.add_argument('--load_model', type=bool, default=False, help="whether use previous model")
net_arg.add_argument('--layer_nums', type=int, default=14, help="nums of hidden layers")
net_arg.add_argument('--c', type=float, default=3, help="constant for tanh layer")
net_arg.add_argument('--mii', type=int, default=2, help="the minimum ii")


net_arg.add_argument('--max_memory', type=float, default=4, help="Number of memory resource in each layer")
net_arg.add_argument('--memory_mode', type=bool, default=True, help="设置memory的打开或者关闭")

net_arg.add_argument('--max_LRF', type=float, default=2, help="Number of PEA registers")
net_arg.add_argument('--max_GRF', type=float, default=2, help="Number of GRF in each layer")

net_arg.add_argument('--reward_mode', type=int, default=1, help="1 is the Mesh structure, 2 is the Torus structure, 3 is the Diagonal structure, and 4 is the 1-HOP structure")


def get_config():
    config, _ = parser.parse_known_args()
    return config
