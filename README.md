# Limit Order Book Event Stream Prediction with Diffusion Model (LOBDIF)

This repository contains the official implementation of the paper:

**"Limit Order Book Event Stream Prediction with Diffusion Model"**

## Overview

We propose a diffusion-based generative framework for modeling and predicting event streams in limit order book (LOB) systems. This approach captures both the irregular temporal dynamics and discrete event types effectively.

## Getting Started

### 1. Run the model

```bash
cd lob-diffusion
python run_model.py --dataset name --samplingsteps 5
```

If you find this work helpful, please consider citing our paper:
@article{zheng2024limitorderbookevent,
    title={Limit Order Book Event Stream Prediction with Diffusion Model}, 
    author={Zetao Zheng and Guoan Li and Deqiang Ouyang and Decui Liang and Jie Shao},
    year={2024},
    eprint={2412.09631},
    archivePrefix={arXiv},
    primaryClass={q-fin.ST},
    url={https://arxiv.org/abs/2412.09631}, 
}


