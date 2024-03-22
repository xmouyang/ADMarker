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








