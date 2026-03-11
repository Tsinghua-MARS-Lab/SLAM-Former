<div align="center">
<h1>SLAM-Former: Putting SLAM into One Transformer</h1>
<a href="https://arxiv.org/abs/2509.16909"><img src="https://img.shields.io/badge/arXiv-2509.16909-b31b1b" alt="arXiv"></a>
<a href="https://tsinghua-mars-lab.github.io/SLAM-Former"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

[Yijun Yuan](https://jarrome.github.io), [Zhuoguang Chen](https://zgchen33.github.io), [Kenan Li](https://connorkevin.github.io), [Weibang Wang](https://William-Wang-1009.github.io), [Hang Zhao](https://hangzhaomit.github.io)

IIIS, Tsinghua University
</div>

```bibtex
@article{slam-former,
      title={SLAM-Former: Putting SLAM into One Transformer}, 
      author={Yijun Yuan, Zhuoguang Chen, Kenan Li, Weibang Wang, and Hang Zhao},
      journal={arXiv preprint arXiv:2509.16909},
      year={2025}
}
```

### Updates
* [Mar 11, 2026] Released training code. See the train branch for details.
* [Mar 4, 2026] Released SLAM code with KV pruning available.
* [Feb 26, 2026] Provides the training data.
* [Sep 24, 2025] Some good blogs can help you read SLAM-Former: [here](https://mp.weixin.qq.com/s/si5EVD1y-1kahadYCx0h8A) and [here](https://zhuanlan.zhihu.com/p/1954116490354721029).
* [Sep 23, 2025] Preprint release.

### Getting Started

#### 1. Clone SLAM-Former
```bash
git clone https://github.com/Tsinghua-MARS-Lab/SLAM-Former.git
cd SLAM-Former
```

#### 2. Create conda environment
```bash
conda create -n SLAM-Former python=3.11
conda activate SLAM-Former 
```

#### 3. Install requirements
```bash
pip install -r requirements.txt
pip install -e .
```

### Running SLAM Demo

Prepare a folder containing your image sequence, then run:

```bash
python slam/demo.py \
    --ckpt_path ckpt/checkpoint.pth.model \
    --image_folder /path/to/your/images/ \
    --output_dir /output/result \
    --target_size 518 \
    --retention_ratio 0.5
```

### Visualization

**Real-time visualization** during inference: add `--vis` to the command above. The 3D reconstruction process can be viewed interactively in [Rerun](https://rerun.io/).

**Static visualization** of saved results:
```bash
python slam/visualize_results.py \
    --result_dir /path/to/output_dir
```

### Data
* **Links:**
  * [Hugging Face](https://huggingface.co/datasets/KevinConnorLee/SLF/tree/main) (ARKitScenes, MVS-Synth, ScanNet)
  * [Hugging Face](https://huggingface.co/datasets/KevinConnorLee/preprocessed_Hypersim/tree/main) (Hypersim)
  * ⏳ [Hugging Face](https://huggingface.co/datasets/KevinConnorLee/SLF/tree/main) (ScanNet++, Blended-MVS, MegaDepth) - *Coming soon*

### Checkpoint
* [Hugging Face](https://huggingface.co/Jarrome/SLAM-Former) — recommended to use `--target_size 518` for inference.
