<div align="center">

# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving

[![arXiv](https://img.shields.io/badge/arXiv-2312.xxxxx-479ee2.svg)](https://arxiv.org/abs/2312.xxxxx)
[![OpenLane-V2](https://img.shields.io/badge/GitHub-OpenLane--V2-blueviolet.svg)](https://github.com/OpenDriveLab/OpenLane-V2)
[![LICENSE](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](./LICENSE)

![lanesegment](figs/lane_segment.jpg "Diagram of Lane Segment")

</div>

> :mailbox_with_mail: Primary contact: [Tianyu Li](https://scholar.google.com/citations?user=X6vTmEMAAAAJ) ( litianyu@opendrivelab.com )

This is the official repository of [**LaneSegNet**: Map Learning with Lane Segment Perception for Autonomous Driving](https://arxiv.org/abs/2312.xxxxx).

## Highlights

:fire: We advocate **Lane Segment** as a map learning paradigm that seamlessly incorporates both map :motorway: geometry and :spider_web: topology information.

:checkered_flag: **Lane Segment** and **`OpenLane-V2 Map Element Bucket`** will serve as a main track in the **`CVPR 2024 Autonomous Driving Challenge`**. For further details, please [stay tuned](https://opendrivelab.com/AD24Challenge.html)!

## News
- **`[2023/12]`** LaneSegNet [paper](https://arxiv.org/abs/2312.xxxxx) is available on arXiv. Code is also released!

---

![method](figs/pipeline.png "Pipeline of LaneSegNet")

<div align="center">
<b>Overall pipeline of LaneSegNet</b>
</div>

## Table of Contents
- [Model Zoo](#model-zoo)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Prepare Dataset](#prepare-dataset)
- [Train and Evaluate](#train-and-evaluate)
- [License and Citation](#license-and-citation)

## Model Zoo

|   Model    | Epoch |  mAP  | TOP<sub>lsls</sub> | Memory | Config | Download |
| :--------: | :---: | :---: | :----------------: | :----: | :----: | :------: |
| LaneSegNet | 24 | 33.5 | 25.4 | 9.4G | [config](projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py) | [model](https://huggingface.co/OpenDrive/lanesegnet_r50_8x1_24e_olv2_subset_A/resolve/main/lanesegnet_r50_8x1_24e_olv2_subset_A.pth)/[log](https://huggingface.co/OpenDrive/lanesegnet_r50_8x1_24e_olv2_subset_A/resolve/main/20231225_213951.log) |


## Prerequisites

- Linux
- Python 3.8.x
- NVIDIA GPU + CUDA 11.1
- PyTorch 1.9.1

## Installation

We recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) to run the code.
```bash
conda create -n lanesegnet python=3.8 -y
conda activate lanesegnet

# (optional) If you have CUDA installed on your computer, skip this step.
conda install cudatoolkit=11.1.1 -c conda-forge

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Install mm-series packages.
```bash
pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.26.0
pip install mmsegmentation==0.29.1
pip install mmdet3d==1.0.0rc6
```

Install other required packages.
```bash
pip install -r requirements.txt
```

## Prepare Dataset

Following [OpenLane-V2 repo](https://github.com/OpenDriveLab/OpenLane-V2/blob/v2.1.0/data) to download the **Image** and the **Map Element Bucket** data. Run following script to collect data for this repo.

```bash
cd LaneSegNet
mkdir data

ln -s {Path to OpenLane-V2 repo}/data/OpenLane-V2 ./data/
python ./tools/data_process.py
```

After setup, the hierarchy of folder `data` is described below:
```
data/OpenLane-V2
├── train
|   └── ...
├── val
|   └── ...
├── test
|   └── ...
├── data_dict_subset_A_train_lanesegment.pkl
├── data_dict_subset_A_val_lanesegment.pkl
├── ...
```

## Train and Evaluate

### Train

We recommend using 8 GPUs for training. If a different number of GPUs is utilized, you can enhance performance by configuring the `--autoscale-lr` option. The training logs will be saved to `work_dirs/lanesegnet`.

```bash
cd LaneSegNet
mkdir -p work_dirs/lanesegnet

./tools/dist_train.sh 8 [--autoscale-lr]
```

### Evaluate
You can set `--show` to visualize the results.

```bash
./tools/dist_test.sh 8 [--show]
```

## License and Citation
All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

If this work is helpful for your research, please consider citing the following BibTeX entry.

``` bibtex
@article{li2023lanesegnet,
  title={LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving},
  author={Li, Tianyu and Jia, Peijin and Wang, Bangjun and Chen, Li and Jiang, Kun and Yan, Junchi and Li, Hongyang},
  journal={arXiv preprint arXiv:2312.xxxxx},
  year={2023}
}

@inproceedings{wang2023openlanev2,
  title={OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping}, 
  author={Wang, Huijie and Li, Tianyu and Li, Yang and Chen, Li and Sima, Chonghao and Liu, Zhenbo and Wang, Bangjun and Jia, Peijin and Wang, Yuting and Jiang, Shengyin and Wen, Feng and Xu, Hang and Luo, Ping and Yan, Junchi and Zhang, Wei and Li, Hongyang},
  booktitle={NeurIPS},
  year={2023}
}
```

## Related resources

We acknowledge all the open-source contributors for the following projects to make this work possible:

- [Openlane-V2](https://github.com/OpenDriveLab/OpenLane-V2)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [TopoNet](https://github.com/OpenDriveLab/TopoNet)