# ADMarker-Example-Datasets

This repo includes examples of real-world multi-modal data collected for Alzheimer’s Disease Monitoring, which are used in the MobiCom 2024 paper: "ADMarker: A Multi-Modal Federated Learning System for Monitoring Digital Biomarkers of Alzheimer’s Disease".



# Download

  The pre-processed example datasets can be downloaded in the [dropbox folder](https://www.dropbox.com/scl/fo/bghugv708agxcmthso9vt/h?rlkey=4pwt0mzqt4bba9159u3bi9bmn&dl=0). For privacy issue, we only released part of collected data from 4 subjects. Please refer to the following discriptions of collecting and pre-processing for each dataset. 
  
  
### Dataset Discriptions: 

* Task: Detect 16 behavior biomarkers in natural home environments, including Dressing, Take/Put something, Cleaning the living area, Grooming, Wiping hands, Drinking, Eating, Phone call/Using phone, Exercising, Talking with others, Stretching, Walking, Sitting, Standing, Lying, Moving in/out of chair. We remove the data samples without humans or from other activities with very limited valid data.
* Sensor Modalities: Depth Camera, mmWave Radar and Microphone.
* Subjects: data from 4 nodes deployed in 4 elderly subjects' homes.
* Size of the dataset: About 1.5 GB.


# Citation
The code and datasets of this project are made available for non-commercial, academic research only. If you would like to use the code or the Alzheimer’s Disease Monitoring datasets of this project, please cite the following papers:
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
