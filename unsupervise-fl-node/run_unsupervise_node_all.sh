CUDA_VISIBLE_DEVICES=0 python3 ./unsupervise_main_node.py --node_id 0 --num_of_samples 3000 --batch_size 16 --epochs 101 --fl_epoch 10 &
CUDA_VISIBLE_DEVICES=1 python3 ./unsupervise_main_node.py --node_id 1 --num_of_samples 3000 --batch_size 16 --epochs 101 --fl_epoch 10 &
CUDA_VISIBLE_DEVICES=2 python3 ./unsupervise_main_node.py --node_id 2 --num_of_samples 3000 --batch_size 16 --epochs 101 --fl_epoch 10 &
CUDA_VISIBLE_DEVICES=3 python3 ./unsupervise_main_node.py --node_id 3 --num_of_samples 3000 --batch_size 16 --epochs 101 --fl_epoch 10 &
