# <p align=center>`PermuMatch: A Trajectory-Centric Framework for Query Permutation Disambiguation in Referring Video Segmentation`</p>

This is the code implementation of the paper "PermuMatch".
## Usage
1. **Environment**

    ```
   Install Python 3.14.3  PyTorch 2.9.1
   cd models/GroundingDINO/ops
   python setup.py build install
   pip install -r requirements.txt
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
   ```
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