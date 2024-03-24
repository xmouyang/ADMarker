CUDA_VISIBLE_DEVICES=0 python3 ./supervise_main_node.py --node_id 0 --batch_size 8 --epochs 101 --fl_epoch 10 &
CUDA_VISIBLE_DEVICES=1 python3 ./supervise_main_node.py -node_id 1 --batch_size 8 --epochs 101 --fl_epoch 10 &
CUDA_VISIBLE_DEVICES=2 python3 ./supervise_main_node.py --node_id 2 --batch_size 8 --epochs 101 --fl_epoch 10 &
CUDA_VISIBLE_DEVICES=3 python3 ./supervise_main_node.py --node_id 3 --batch_size 8 --epochs 101 --fl_epoch 10 &
