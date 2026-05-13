# <p align=center>`PermuMatch: A Trajectory-Centric Framework for Query Permutation Disambiguation in Referring Video Segmentation`</p>

This is the code implementation of the paper "PermuMatch".
## Abstract
Referring Video Object Segmentation (RVOS) aims to segment a language-described object throughout a video. Existing query‑based methods often suffer from query permutation ambiguity due to the permutation‑equivariant nature of DEtection TRansformer (DETR)‑like decoders, which breaks object identity across frames and leads to temporal inconsistency under challenging conditions such as occlusion or similar distractors. To overcome this, we propose PermuMatch, a trajectory‑centric framework that explicitly recovers a temporally coherent and language‑aligned object trajectory. PermuMatch consists of three key components. First, a Hierarchical Optimal Transport‑guided structural Matching module (HOTM) aligns queries across frames by first pruning candidates with coarse feature similarity, then applying fine‑grained Sinkhorn OT on local ROI features within the candidate regions, yielding robust structural correspondences. Second, a Diffusion Trajectory Refinement module (DTR) treats the aligned sequence as a noisy trajectory, denoises it under text conditioning using a lightweight diffusion model, and updates the query features with the denoised result to resolve residual noise and matching ambiguity. Third, a Memory‑to‑Object (M2O) relation distillation loss transfers the stable correspondence structure from the refined trajectory to the aligned query features at the level of relation matrices, avoiding the pitfalls of direct feature distillation and strengthening identity preservation. All modules are trained jointly with a unified loss, forming a closed loop from alignment to refinement to distillation. Experiments on three RVOS benchmarks (Ref-YouTube-VOS, Ref-DAVIS17, and MeViS) demonstrate that PermuMatch establishes consistent cross-frame object correspondence and achieves state-of-the-art performance among query-based methods. Especially, on the challenging MeViS, PermuMatch achieves 49.6% in J&F, improving J&F by 3.9 points over a strong baseline.
## Usage

1. **Environment**

    ```
   Install Python 3.14.3  PyTorch 2.9.1
   cd models/GroundingDINO/ops
   python setup.py build install
   pip install -r requirements.txt
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

2. **Datasets preparation**

   The MEVIS dataset can be downloaded [here](https://github.com/henghuiding/MeViS), Ref-Youtube can be downloaded [here](https://youtube-vos.org/dataset/rvos/), and Ref-DAVIS17 can be downloaded [here](https://davischallenge.org/davis2017/code.html).

   The directory structure of the dataset is as follows:

    ```
    +-- PermuMatch
    |   ...
    |   +-- data
    |       +-- mevis
    |           +-- train
    |               +-- JPEGImages
    |               +-- mask_dict.json
    |               +-- meta_expressions.json
    |           +-- valid_u
    |               +-- JPEGImages
    |               +-- mask_dict.json
    |               +-- meta_expressions.json
    |           +-- valid
    |               +-- JPEGImages
    |               +-- meta_expressions.json
    |       +-- ref_youtube_vos
    |               +-- meta_expressions
    |               +-- train
    |                   +-- Annotations
    |                   +-- JPEGImages
    |                   +-- meta.json
    |               +-- valid
    |                   +-- Annotations
    |                   +-- JPEGImages
    |                   +-- meta_expressions_challenge.json
    |       +-- ref_davis
    |               +-- meta_expressions
    |               +-- davis_text_annotations
    |               +-- train
    |                   +-- Annotations
    |                   +-- JPEGImages
    |                   +-- meta.json
    |               +-- valid
    |                   +-- Annotations
    |                   +-- JPEGImages
    |                   +-- meta.json
    |   ...
    ```
3. **Inference**

   Download model weights: [MeViS](https://pan.baidu.com/s/1_33Z85H5b8VkLpQEdQNHgQ).

   Run the following command to perform inference on MeViS.
   ```
   python eval/inference_mevis.py --split valid -c configs/mevis_swinb.yaml -ng 8 ckpt/mevis_swinb.pth --version swinb
   ```
   After obtaining the inference results, submit the compressed archive to the evaluation server [MeViS Server](https://www.codabench.org/competitions/12222/).

4. **Acknowledgements**

   The code is based on [MeViS](https://github.com/henghuiding/MeViS), [ReferFormer](https://github.com/wjn922/ReferFormer), [SOC](https://github.com/RobertLuo1/NeurIPS2023_SOC), [ReferDINO](https://github.com/iSEE-Laboratory/ReferDINO) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and we are very grateful for their valuable work.