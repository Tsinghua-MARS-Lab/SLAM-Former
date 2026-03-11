# SLAM-Former Training Code

This repository contains the training code for [SLAM-Former](https://github.com/Tsinghua-MARS-Lab/SLAM-Former).

## Environment Setup & Data Preparation

Please refer to the [main README](https://github.com/Tsinghua-MARS-Lab/SLAM-Former) for environment setup and data download instructions.

After downloading, update the dataset root paths in `config/mytrain.yaml`

Also set the pretrained checkpoint path:

```yaml
pretrained: /your/path/to/pi3.pth
```

## Training

We use [Accelerate](https://github.com/huggingface/accelerate) for distributed training. Configure your setup with:

```bash
accelerate config
```

### Single-GPU

```bash
cd src
accelerate launch --num_processes=1 mytrain.py
```

### Multi-GPU (single node)

```bash
cd src
accelerate launch --num_processes=8 --multi_gpu mytrain.py
```


## Acknowledgements

This codebase is built on top of the following excellent projects:

- [DUSt3R](https://github.com/naver/dust3r) — the foundational 3D reconstruction framework we build upon
- [StreamVGGT](https://github.com/wzzheng/StreamVGGT) — streaming architecture and training utilities
- [VGGT](https://github.com/facebookresearch/vggt) — geometry and pose encoding

We sincerely thank the authors of these works for open-sourcing their code.

## Citation

If you find this work useful, please cite:

```bibtex
@article{slam-former,
    title={SLAM-Former: Putting SLAM into One Transformer},
    author={Yijun Yuan, Zhuoguang Chen, Kenan Li, Weibang Wang, and Hang Zhao},
    journal={arXiv preprint arXiv:2509.16909},
    year={2025}
}
```
