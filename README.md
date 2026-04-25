# Reinforcing Diffusion Models by Direct Group Preference Optimization

This is the Official Repository of "[Reinforcing Diffusion Models by Direct Group Preference Optimization](https://arxiv.org/pdf/2510.08425)", by *Yihong Luo, Tianyang Hu, Jing Tang*.

<table align="center">
  <tr>
    <td align="center" width="100%"><img src="teaser.png" alt="Teaser"></td>
  </tr>
  <tr>
    <td colspan="2" align="center">Our proposed <b>DGPO</b> shows a near 30 times faster training compared to Flow-GRPO on improving GenEval score (Left Figure). The notable improvement is achieved while maintaining strong performance on other out-of-domain metrics (Right Figure).</td>
  </tr>
</table>

## Table of Contents

- [News](#news)
- [Method](#method)
- [Main Results](#main-results)
- [Quick Started](#-quick-started)
  - [Reward Preparation](#reward-preparation)
  - [Start Training](#start-training)
  - [Hyper-parameter Recipes](#hyper-parameter-recipes)
    - [Core Loss Coefficients](#core-loss-coefficients)
    - [Mode 1 — Rollout w/ CFG, training w/o CFG (default)](#mode-1--rollout-w-cfg-training-wo-cfg-default)
    - [Mode 2 — Fully CFG-free](#mode-2--fully-cfg-free)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)
- [Bibtex](#bibtex)

## 🔥News
- (2026/04) DGPO has been integrated into [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory).
- (2026/01) DGPO is accepted to ICLR 2026 🎉!
- (2026/03/07) Code is ready to be released next week.
- (2026/03/14) Training code is released.

## Method
The key insight of our work is that the success of methods like GRPO stems from leveraging fine-grained, relative preference information within a group of samples, not from the policy-gradient formulation itself. Existing methods for diffusion models force the use of inefficient stochastic (SDE) samplers to fit the policy-gradient framework, leading to slow training and suboptimal sample quality.

**DGPO** circumvents this problem by optimizing group-level preferences directly, *extending the Direct Preference Optimization (DPO) framework to handle **pairwise groups** instead of pairwise samples*. This allows us to:
- **Use Efficient Samplers:** Employ fast and high-fidelity deterministic ODE samplers for generating training data, leading to better-quality rollouts.
- **Learn Directly from Preferences:** Optimize the model by maximizing the likelihood of group-wise preferences, eliminating the need for a stochastic policy and inefficient random exploration.
- **Train Efficiently:** Avoid training on the entire sampling trajectory, significantly reducing the computational cost of each iteration.

For a group of generated samples, we partition them into a positive group $\mathcal{G}^+$ and a negative group $\mathcal{G}^-$ based on their reward. The model with parameter $\theta$ is trained by maximum likelihood learning of a group-level reward: 

$$
\max_{\theta} E_{(\mathcal{G}^+, \mathcal{G}^-, c) \sim \mathcal{D}} \log p_\theta(\mathcal{G}^+ \succ \mathcal{G}^-|c) = E_{(\mathcal{G}^+, \mathcal{G}^-, c) \sim \mathcal{D}} \log\sigma(R_\theta(\mathcal{G}^+|c) - R_\theta(\mathcal{G}^-|c)).
$$

We refer to our paper for more details.

## Main Results

DGPO consistently outperforms Flow-GRPO on target metrics for Compositional Generation, Text Rendering, and Human Preference Alignment, while also showing strong or superior performance on out-of-domain quality and preference scores.

| Model | GenEval | OCR Acc. | PickScore | Aesthetic | DeQA | ImgRwd | PickScore | UniRwd |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SD3.5-M | 0.63 | 0.59 | 21.72 | 5.39 | 4.07 | 0.87 | 22.34 | 3.33 |
| **_Compositional Image Generation:_** | | | | | | | | |
| Flow-GRPO | 0.95 | --- | --- | 5.25 | 4.01 | 1.03 | 22.37 | 3.51 |
| **DGPO (Ours)** | **0.97** | --- | --- | **5.31** | **4.03** | **1.08** | **22.41** | **3.60** |
| **_Visual Text Rendering:_** | | | | | | | | |
| Flow-GRPO | --- | 0.92 | --- | 5.32 | 4.06 | 0.95 | 22.44 | 3.42 |
| **DGPO (Ours)** | --- | **0.96** | --- | **5.37** | **4.09** | **1.02** | **22.52** | **3.48** |
| **_Human Preference Alignment:_** | | | | | | | | |
| Flow-GRPO | --- | --- | 23.31 | 5.92 | 4.22 | 1.28 | 23.53 | 3.66 |
| **DGPO (Ours)** | --- | --- | **23.89** | **6.08** | **4.40** | **1.32** | **23.91** | **3.74** |


## 🚀 Quick Started

Clone this repository and install packages.
```bash
git clone https://github.com/Luo-Yihong/DGPO.git
cd DGPO
conda create -n flow_grpo python=3.10.16
pip install -e .
```
### Reward Preparation
```bash
# GenEval
pip install -U openmim
mim install mmengine
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv; git checkout 1.x
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e . -v
cd ..

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -e . -v
cd ..

# OCR
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein
```


### Start Training

***By default, DGPO uses the "CFG during inference, no CFG during training" mode.*** We found that this setup demonstrates a better trade-off between training time and the OOD metric.

```bash
bash scripts/single_node/sd3_dgpo_ocr.sh
```

For other modes (fully CFG-free with frozen / dynamic reference) and the recommended hyper-parameters, see [Hyper-parameter Recipes](#hyper-parameter-recipes) below.

### Hyper-parameter Recipes

#### Core Loss Coefficients

DGPO is controlled by a small set of keys spread across `config.train.*`, `config.sample.*`, and the top-level `config.*` namespace (see [config/base.py](config/base.py) and per-task configs in [config/](config/)):

| Key | Meaning |
| --- | --- |
| `config.train.beta_dpo` | DPO beta scaling for group preference; larger -> sharper sigmoid weighting. |
| `config.train.beta` | KL penalty weight (a.k.a. `kl_beta`). 0 disables the KL term. |
| `config.kl_cfg` | CFG scale on the (frozen) reference. >1 enables CFG on the KL reference branch. |
| `config.sample.guidance_scale` | CFG used during the rollout process. |
| `config.clip_range` | PPO-style clip range (scalar, expanded to `(-c, c)`). |
| `config.use_ema_ref` | If `True`, use a dynamic EMA reference model ([TDM-R1](https://arxiv.org/abs/2603.07700) style). |

DGPO supports two practical modes: (1) rollout w/ CFG, training w/o CFG; (2) CFG-free in both rollout and training.

#### Mode 1 — Rollout w/ CFG, training w/o CFG (default)

Best trade-off between training time and OOD performance. The reference model is frozen and used **without** CFG. Typical ranges: `beta_dpo` 10 ~ 100, `clip_range` 1e-3 ~ 1e-2.

```bash
bash scripts/single_node/sd3_dgpo_ocr.sh
```

```python
# excerpt from config/dgpo.py
config.train.beta_dpo        = 100.0
config.train.beta            = 1e-3
config.kl_cfg                = 1.0
config.sample.guidance_scale = 4.5
config.clip_range            = 1e-3
```

#### Mode 2 — Fully CFG-free

CFG-free converges significantly faster but generally trades off some OOD performance. A **small** PPO clip is recommended for stability: `clip_range` 1e-5 ~ 1e-4. Two reference-model choices (using a reference w/o CFG slows training down with large `beta_dpo`, or makes it unstable with small `beta_dpo`, so it is not recommended):

**(a) Frozen reference with CFG on the KL branch** — `beta_dpo` 10 ~ 100:

```bash
bash scripts/single_node/sd3_dgpo_ocr_wocfg.sh
```

```python
# excerpt from config/dgpo_wocfg.py
config.train.beta_dpo        = 100.0
config.train.beta            = 1e-3
config.kl_cfg                = 4.5
config.sample.guidance_scale = 1.0
config.clip_range            = 1e-5
config.use_ema_ref           = False
```

**(b) Dynamic EMA reference model**  — `beta_dpo` typically larger, 2000 ~ 5000:

```bash
bash scripts/single_node/sd3_dgpo_ocr_wocfg_emaref.sh
```

```python
# excerpt from config/dgpo_wocfg_emaref.py
config.train.beta_dpo        = 2000.0
config.train.beta            = 1e-3
config.kl_cfg                = 1.0 # or 4.5/3.5
config.sample.guidance_scale = 1.0
config.clip_range            = 1e-5
config.use_ema_ref           = True
```



## Acknowledgement
Motivated by a thoughtful comment during the review process, we have added PPO-style clipping in training, where we observed improved training stability. Thanks to the effort by the anonymous reviewers. Our codebase is largely built upon [Flow-GRPO](https://github.com/yifan123/flow_grpo). We simplify the environment setup for GenEval following [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT). We thank the authors for their efforts to the open-source codebase.


## Contact

Please contact Yihong Luo (yluocg@connect.ust.hk) if you have any questions about this work.

## Bibtex

```
@inproceedings{luo2026dgpo,
              title={Reinforcing Diffusion Models by Direct Group Preference Optimization},
              author={Yihong Luo and Tianyang Hu and Jing Tang},
              booktitle={The Fourteenth International Conference on Learning Representations},
              year={2026},
}
```

