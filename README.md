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
	<img src="https://github.com/xmouyang/ADMarker/blob/main/figure/system-overview.png" width="800">
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

    |-- run_unsupervise_node.sh/	// run unsupervised FL of a client on a edge device
    |-- unsupervise_main_node.py/	// main file of running unsupervised FL on the client
    |-- communication.py/	// set up communication with server
    |-- data_pre.py/		// load the data for clients in FL
    |-- cosmo_design.py/	// contrastive fusion learning
    |-- model.py/ 	// model configurations
    |-- util.py		// utility functions

|--unsupervise-fl-server // codes running unsupervised FL on the server
    |-- unsupervise_main_server.py

|--supervise-fl-node // codes running supervised FL on clients

    |-- run_supervise_node.sh/	// run supervised FL of a client on a edge device
    |-- supervise_main_node.py/	// main file of running supervised FL on the client
    |-- communication.py/	// set up communication with server
    |-- data_pre.py/		// load the data for clients in FL
    |-- model.py/ 	// model configurations
    |-- util.py		// utility functions

|--supervise-fl-server // codes running supervised FL on the server
    |-- supervise_main_server.py

```
<br>

# Quick Start 
* Download the codes for each dataset in this repo. Put the folder `unsupervise-fl-node` and `supervise-fl-node` on your client machines, and `unsupervise-fl-server` and `supervise-fl-server` on your server machine.
* Download the `dataset` from [ADMarker-Example-Datasets](https://github.com/xmouyang/ADMarker/blob/main/dataset.md) to your client machines. Put the folder `under the same folder` with codes of running FL on clients. You can also change the path of loading datasets in 'data_pre.py' to the data path on your client machine.
* Download the `pretrain_model.pth` from [pre-trained model weights](https://github.com/xmouyang/ADMarker/blob/main/dataset.md) to your client machines. Put the folder `under the same folder` with codes of running FL on clients. 
* Change the argument "server_address" in 'unsupervise_main_node.py' and 'supervise_main_node.py' as your true server address. If your server is located in the same physical machine of your nodes, you can choose "localhost" for this argument.
* Run unsupervised federated learning on the clients and server:
	* Server:
		```bash
	    python3 unsupervise_main_server.py
	    ```
	* Client: change the 'node_id' (0,1, 2, ...) in the below script for each client
	    ```bash
	    ./run_unsupervise_node.sh
	    ```
* Run supervised federated learning on the clients and server:
	* Server:
		```bash
	    python3 supervise_main_server.py
	    ```
	* Client: change the 'node_id' (0,1, 2, ...)  in the below script for each client
	    ```bash
	    ./run_supervise_node.sh
	    ```
* NOTE: The default codes corresponde to the settings with four nodes, as we only released data from four subjects due to the privacy concerns. If you want to adapt the codes to other datasets with more nodes, you should change the hyper-parameter `num_of_users` in `unsupervise_main_server.py` and `unsupervise_main_server.py`, as well as the `node_id` in `run_unsupervise_node.sh` and `run_supervise_node.sh `




