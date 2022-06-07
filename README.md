# SWEM: Towards Real-Time Video Object Segmentation with Sequential Weighted Expectation-Maximization

This repository is the official implementation of SWEM: Towards Real-Time Video Object Segmentation with Sequential Weighted Expectation-Maximization (CVPR2022)

![](https://github.com/lmm077/SWEM/blob/main/assets/pipeline.pdf)

## 1. Requirements

We use  one NVIDIA V100 card (16 GB Memory), two 1080ti GPUs are satisfied, if you use one 1080ti, you can reduce the batch size and increase number of iterations.
To install requirements, run:

```bash
pip3 install -r requirements.txt
```

## 2. RUN

### Datasets

#### Image Data

You can download and process image datasets from [STCN](https://github.com/hkchengrex/STCN) or directly download from [Google Drive](https://drive.google.com/file/d/12hfHQ5cBflEearH7rWiJs6q-B00UyQhx/view?usp=sharing)

#### Video Data

You need download the [DAVIS17-TrainVal](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) and  [YouTube-VOS](https://youtube-vos.org/) datasets for main-training and testing. 

You need modified the data path in configs/config.py

### Training and Testing

Main training and testing

```
sh train_swem_s3.sh
```

## 3. License

This repository is released for academic use only. If you want to use our codes for commercial products, please contact linchrist@163.com in advance.

## 4. Related Repos

https://github.com/seoungwugoh/STM

https://github.com/haochenheheda/Training-Code-of-STM

https://github.com/hkchengrex/STCN

Codes of data samplers are from https://github.com/dvlab-research/Simple-SR
