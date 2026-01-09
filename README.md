<h1 align="center" style="margin-top: 10px;">AT<sup>2</sup>PO: Agentic Turn-based Policy Optimization via Tree Search</h1>



<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2601.04767)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/papers/2601.04767)
<!-- [![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn/@yux1ang/Tree-GRPO/overview) -->

</div>

<!-- ## News
- [Sep 25, 2025]: Codebase released. (work in progress) -->

## Table of contents

- [Overview](#overview)
- [Quick start](#quick-start)
- [Preliminary results](#preliminary-results)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


## Overview
we present **AT<sup>2</sup>PO** (**A**gentic **T**urn-based **P**olicy **O**ptimization via **T**ree Search), a unified framework for multi-turn agentic RL that addresses three core challenges: limited exploration diversity, sparse credit assignment, and misaligned policy optimization. AT<sup>2</sup>PO introduces a turn-level tree structure that jointly enables (a) **Entropy-Guided Tree Expansion** for strategic exploration and (b) **Turn-wise Credit Assignment** for fine-grained reward propagation from sparse outcomes. Complementing this, we propose (c) **Agentic Turn-based Policy Optimization**, a turn-level learning objective that aligns policy updates with the natural decision granularity of agentic interactions. 

<p align="center">
  <img alt="intro" src="assets/framework.png" />
  <i>
  The overview of AT<sup>2</sup>PO framework.
  </i>
</p>

*Evaluation on seven benchmarks shows consistent improvement against existing strongest baselines.*

## Quick Start

### Local Retriever Tool Initialization

#### Environment
```bash
conda create -y -n retriever python=3.10
conda activate retriever
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

#### Download Retriever Data
```bash
save_path=/the/path/to/save
python rag_server/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

#### Initialize Retriever API
```bash
conda activate retriever
# edit save_path in rag_server/launch.sh
bash rag_server/launch.sh
```

### Dataset
```bash
# Process training set of multi-hop QA benchmarks
python data_process/hotpotqa_multihop_train.py
# Process test set of multi-hop QA benchmarks
python data_process/multihop_test_merge.py
# Process training set of single-hop QA benchmarks
python data_process/nq_singlehop_train.py
# Process test set of single-hop QA benchmarks
python data_process/singlehop_test_merge.py
```


### Training Environment Installation

```bash
#create env
conda create -n atpo python==3.10
conda activate atpo

# install torch & flash-atten
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation

# install RL basic env
cd ATPO

# This is our RL env freeze file. You can install it as a supplement or use it for checking.
pip install -r requirements.txt

```

### RL Training

Run GRPO training with Qwen3-4B on multi-hop QA setting.
```bash
conda activate atpo
bash AEPO/scripts/GRPO_qwen3_4B.sh
```

Run AEPO training with Qwen3-4B on multi-hop QA setting.
```bash
conda activate atpo
bash AEPO/scripts/AEPO_qwen3_4B.sh
```

Run AT<sup>2</sup>PO training with Qwen3-4B on multi-hop QA setting.
```bash
conda activate atpo
bash ATPO/scripts/ATPO_qwen3_4B.sh
```

## Evaluation results



|        | **Hotpot** | **2wiki** | **Musiq** | **Bamb** | **Avg.** | **NQ** | **TriviaQA** | **PopQA** | **Avg.** |
|--------|-------------|---------|--------|------|------|-----|-----------|-------|------|
| **Backbone Model: Qwen3-4B** | | | | | | | | | |
| ReAct | 30.42 | 32.92 | 12.83 | 44.80 | 30.01 | 26.75 | 53.53 | 35.34 | 41.31 |
| + GRPO | 44.76 | 51.40 | 21.60 | 50.40 | 46.02 | 45.98 | 65.17 | 49.18 | 54.97 |
| + DAPO | 45.95 | 51.81 | 21.68 | 51.20 | 46.65 | 47.50 | **65.84** | 51.03 | 56.33 |
| + GSPO | 47.07 | 49.25 | 22.68 | 50.40 | 45.69 | 46.01 | 64.24 | 48.50 | 54.28 |
| + AEPO | 46.36 | 51.78 | 23.47 | 50.40 | 46.95 | 45.71 | 64.66 | 50.13 | 55.20 |
| + **AT²PO (Ours)** | **49.44** | **52.99** | **24.80** | **56.80** | **48.81** | **47.90** | 65.32 | **51.81** | **56.44** |
| **Backbone Model: Qwen3-8B** | | | | | | | | | |
| ReAct | 20.66 | 19.05 | 9.56 | 37.60 | 18.66 | 21.16 | 41.81 | 27.37 | 32.19 |
| + GRPO | 47.01 | 53.69 | 21.35 | 54.40 | 48.03 | 45.70 | 67.42 | 50.17 | 56.29 |
| + DAPO | 49.64 | 53.91 | 24.05 | 56.00 | 49.40 | **51.99** | 69.02 | 51.90 | 58.53 |
| + GSPO | 49.59 | 52.55 | 24.35 | 54.40 | 48.56 | 45.56 | 67.75 | 49.66 | 56.15 |
| + AEPO | 49.17 | 52.97 | 24.01 | 54.40 | 48.62 | 49.92 | 68.31 | 51.77 | 57.94 |
| + **AT²PO (Ours)** | **51.37** | **53.97** | **26.51** | **56.00** | **50.15** | 51.33 | **69.51** | **52.26** | **58.82** |
| **Backbone Model: Qwen2.5-7B** | | | | | | | | | |
| ReAct | 2.85 | 1.94 | 0.58 | 4.00 | 2.10 | 4.34 | 10.67 | 9.32 | 9.23 |
| + GRPO | 47.94 | 46.89 | 21.27 | 47.20 | 44.48 | 45.56 | 64.86 | 49.92 | 55.20 |
| + DAPO | 47.50 | 47.93 | 21.27 | 44.00 | 44.91 | 52.24 | **65.00** | 50.01 | 56.08 |
| + GSPO | 47.35 | 47.30 | 20.32 | 44.00 | 44.40 | 49.64 | 62.87 | 49.75 | 54.81 |
| + Tree-GRPO | 42.39 | 42.01 | 20.15 | 42.40 | 39.79 | 47.56 | 62.69 | 44.75 | 52.04 |
| + AEPO | 47.05 | 47.53 | 21.03 | 44.00 | 44.51 | 49.00 | 64.13 | 50.21 | 55.45 |
| + **AT²PO (Ours)** | **49.58** | **48.04** | **22.56** | **51.20** | **45.83** | **52.91** | 64.90 | **50.44** | **56.34** |

**Table Caption:** Experiment results on three backbone models across seven datasets. The **bolded** values indicate the best result in comparisons.


## Acknowledgement
The codebase is built upon [veRL](https://github.com/volcengine/verl).
The implementation is inspired by [AEPO](https://github.com/RUC-NLPIR/ARPO).
We express our gratitude to these open-source projects.

## Citation
```bibtex
@misc{zong2026at2poagenticturnbasedpolicy,
      title={AT$^2$PO: Agentic Turn-based Policy Optimization via Tree Search}, 
      author={Zefang Zong and Dingwei Chen and Yang Li and Qi Yi and Bo Zhou and Chengming Li and Bo Qian and Peng Chen and Jie Jiang},
      year={2026},
      eprint={2601.04767},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.04767}, 
}
```