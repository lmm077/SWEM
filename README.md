# SWEM: Towards Real-Time Video Object Segmentation with Sequential Weighted Expectation-Maximization

This repository is the official implementation of SWEM: Towards Real-Time Video Object Segmentation with Sequential Weighted Expectation-Maximization (CVPR2022)

![](https://github.com/lmm077/SWEM/blob/main/assets/pipeline.png)


## 1. Requirements

- We use  one NVIDIA V100 (16 GB Memory), whereas two 1080ti GPUs are also satisfied. Note that if you use one 1080ti, you can reduce the batch size and increase number of iterations correspondingly.

- To install requirements, run:

```bash
pip3 install -r requirements.txt
```

## 2. Preparing datasets

- Image Data: Download and process image datasets from [STCN](https://github.com/hkchengrex/STCN) or directly download from [Google Drive](https://drive.google.com/file/d/12hfHQ5cBflEearH7rWiJs6q-B00UyQhx/view?usp=sharing)

- Video Data: Download the [DAVIS17-TrainVal](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) and  [YouTube-VOS](https://youtube-vos.org/) datasets for main-training and testing. 

- Modify the data path in configs/config.py

## 3. Training and Testing

- Main training and testing

```
sh train_swem_s3.sh
```

## 4. License

This repository is released for academic use only. If you want to use our codes for commercial products, please contact linchrist@163.com in advance.

## 5. Related Repos

https://github.com/seoungwugoh/STM

https://github.com/haochenheheda/Training-Code-of-STM

https://github.com/hkchengrex/STCN

Codes of data samplers are from https://github.com/dvlab-research/Simple-SR

## 6. Citation
```
  @inproceedings{SWEM,
  title={SWEM: Towards Real-Time Video Object Segmentation with Sequential Weighted Expectation-Maximization},
  author={Lin, Zhihui and Yang, Tianyu and Li, Maomao and Wang, Ziyu and Yuan, Chun and Jiang, Wenhao and Liu, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1362--1372},
  year={2022}
  }
```
