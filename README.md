# ADMarker
This is the repo for MobiCom 2024 paper: "ADMarker: A Multi-Modal Federated Learning System for Monitoring Digital Biomarkers of Alzheimer’s Disease".

# Citation
The code and datasets of this project are made available for non-commercial, academic research only. If you would like to use the code or datasets of this project, please cite the following papers:
```
@article{ouyang2023admarker,
  title={ADMarker: A Multi-Modal Federated Learning System for Monitoring Digital Biomarkers of Alzheimer's Disease},
  author={Ouyang, Xiaomin and Shuai, Xian and Li, Yang and Pan, Li and Zhang, Xifan and Fu, Heming and Wang, Xinyan and Cao, Shihua and Xin, Jiang and Mok, Hazel and others},
  journal={arXiv preprint arXiv:2310.15301},
  year={2023}
}
@article{ouyang2024admarker,
  title={ADMarker: A Multi-Modal Federated Learning System for Monitoring Digital Biomarkers of Alzheimer's Disease},
  author={Ouyang, Xiaomin and Shuai, Xian and Li, Yang and Pan, Li and Zhang, Xifan and Fu, Heming and Wang, Xinyan and Cao, Shihua and Xin, Jiang and Mok, Hazel and others},
  journal={Proceedings of the 30th Annual International Conference on Mobile Computing And Networking},
  year={2024}
}
```
# Requirements
The program has been tested in the following environment:
* Computing Clusters: Python 3.9.7, Pytorch 1.12.0, torchvision 0.13.0, CUDA Version 10.2, sklearn 0.24.2, numpy 1.20.3
* Nvidia Xavier NX: Ubuntu 18.04.6, Python 3.6.9, Pytorch 1.8.0, CUDA Version 10.2, sklearn 0.24.2, numpy 1.19.5
<br>

# ADMarker FL Overview
<p align="center" >
	<img src="https://github.com/xmouyang/ADMarker/blob/main/figure/three-stage-framework.png" width="800">
</p>

First Stage: Centralized model pre-training

Second Stage: Unsupervised multi-modal federated learning
* Client: 
	* Local unsupervised multimodal training with contrastive fusion learning
	* Send model weights to the server.
* Server: 
	* Aggregate model weights of different modalities with Fedavg;
	* Send the aggregated model weights to each client.
	
Third Stage: Supervised multi-modal federated learning
* Client: 
	* Local fusion: train the classifier layers with labeled data;
	* Send model weights to the server.
* Server: 
	* Send the aggregated model weights to each client.


# Project Strcuture
```
|--unsupervise-fl-node // codes running unsupervised FL on clients

    |-- run_unsupervise_node_all.sh/	// run unsupervised FL of all clients on a cloud cluster
    |-- run_unsupervise_node.sh/	// run unsupervised FL of a client on a edge device
    |-- unsupervise_main_node.py/	// main file of running unsupervised FL on the client
    |-- communication.py/	// set up communication with server
    |-- data_pre.py/		// load the data for clients in FL
    |-- cosmo_design.py/	// contrastive fusion learning
    |-- model.py/ 	// model configurations
    |-- util.py		// utility functions

|--unsupervise-fl-server // codes running unsupervised FL on the server

|--supervise-fl-node // codes running supervised FL on clients

|--supervise-fl-server // codes running supervised FL on the server

```
<br>

# Quick Start 
* Download the codes for each dataset in this repo. Put the folder `client` on your client machines and `server` on your server machine.
* Download the `dataset` (three public datasets and one dataset collected by ourselves for AD monitoring) from [Harmony-Datasets](https://github.com/xmouyang/Harmony/blob/main/dataset.md) to your client machines.
* Choose one dataset from the above four datasets and put the folder `under the same folder` with corresponding codes. You can also change the path of loading datasets in 'data_pre.py' to the data path on your client machine.
* Change the argument "server_address" in 'main_unimodal.py' and 'main_fedfuse.py' as your true server address. If your server is located in the same physical machine of your nodes, you can choose "localhost" for this argument.
* Run the following code on the client machine
	* For running clients on the cloud cluster (clients are assigned to different GPUs)
		* Run the first stage
		    ```bash
		    ./run_unifl_all.sh
		    ```
		* Run the second stage
		    ```bash
		    ./run_fedfusion_all.sh
		    ```
		* NOTE: You may need to change the running scripts "run_unifl_all.sh" and "run_fedfusion_all.sh" if you want to run multiple nodes on the same GPUs or run the nodes on different machines. For example, if you want to run 14 nodes in the USC dataset on only 4 GPUs, please run the shell scripts "run_unifl_all-4GPU.sh" and "run_fedfusion_all-4GPU.sh"; if you want to run 16 nodes in the self-collected AD dataset on 4 different machines, please move the shell scripts from the folder "node-run-stage1-4cluster" and "node-run-stage1-4cluster" to the source folder and run one script on each machine.
	* For running clients on the edge devices (clients are assigned to different Nvidia Xavier NX device)
		* Move the running script of each node (run_unifl_xx.sh, run_unifl_schedule_xx.sh and run_fedfusion_xx.sh) from the folder 'node-run-stage1' and 'node-run-stage2' to the folder 'client'
		* Run the first stage: 
			* For single-modal nodes: 
			    ```bash
			    ./run_unifl_xx.sh
			    ```
			* For multi-modal nodes: 
			    ```bash
			    ./run_unifl_schedule_xx.sh
			    ```
		* Run the second stage: 
		    ```bash
		    ./run_fedfusion_xx.sh
		    ```
* Run the following code on the server machine
	* Run the first stage: run multiple tasks for different unimodal FL subsystems
	    ```bash
	    python3 main_server_stage1_uniFL.py --modality_group 0
	    python3 main_server_stage1_uniFL.py --modality_group 1
	    python3 main_server_stage1_uniFL.py --modality_group 2
	    ```
	* Run the second stage
	    ```bash
	    python3 main_server_stage2_fedfusion_3modal.py
	    ```






