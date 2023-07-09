### Instructions for use
This code runs under Ubuntu 18.04.6 with tensorflow2.6.0

If you want to run the code, enter it on the command line:
bash run.sh demo

If you want to modify the parameters of the code, open the run.sh file and modify the specified parameter information
The following is an explanation of some key parameters
1. src_file_path 	#For example, data/o22m.txt is the 22nd kernel
2. actor_lr 	#Representation learning rate
3. gcn_dims	#Indicates the hidden layer dimension
4. max_iteration	#Indicates the maximum number of iterations
5. batch_size	#Represents the number of samples in a batch
6. pea_width	#Indicates the scale of the CGRA. For example, 6 indicates that the scale of the CGRA is 6x6
7. reward_mode	#Indicates the CGRA structure, 1 indicates the mesh structure, 2 indicates the torus structure, 3 indicates the Diagonal structure, and 4 indicates the 1-HOP structure
8. max_LRF	#Indicates that there are several LRF resources in one FU
9. max_memory	#Indicates that there are several LSU resources in a time slot
10. max_GRF	#Indicates several GRF resources in a time slot

