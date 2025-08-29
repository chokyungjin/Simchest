# SimChest: a novel anomaly agnostic model for similarity measurement in follow-up chest radiograph pairs via a supervised contrastive learning model

This is a PyTorch implementation of the [SimChest paper]((https://link.springer.com/article/10.1007/s10278-024-01245-0)):
```
@article{kim2025screening,
  title={Screening Patient Misidentification Errors Using a Deep Learning Model of Chest Radiography: A Seven Reader Study},
  author={Kim, Kiduk and Cho, Kyungjin and Eo, Yujeong and Kim, Jeeyoung and Yun, Jihye and Ahn, Yura and Seo, Joon Beom and Hong, Gil-Sun and Kim, Namkug},
  journal={Journal of Imaging Informatics in Medicine},
  volume={38},
  number={2},
  pages={694--702},
  year={2025},
  publisher={Springer}
}
```

## Pretraining task procedure

![스크린샷 2023-05-23 오후 12 40 56](https://github.com/chokyungjin/Bi_rads/assets/108312461/aaf8e058-a71d-409c-abac-c9248fff92e6)


## Model training

```

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_supcon.py --dataset real --name SupCon_scratch \
--print_freq=5 --save_freq 1 --num_workers 8 --aug True --warm \
--batch_size 8 --model resnet50 --method SupCon --epochs 100

```

## Pretrained model weight
[Google Drive](https://drive.google.com/uc?id=1I8IgZ6mjPwvCPDk0EutkYH6UYOY865_n&export=download)
or you can use gdown in Python

```python
!gdown https://drive.google.com/uc?id=1I8IgZ6mjPwvCPDk0EutkYH6UYOY865_n
```

## Model inference

If you want to get CXR image similarity logit, run the code below.

```
python model_inference.py

```


## Result

![Figure_CAM](https://github.com/chokyungjin/Bi_rads/assets/108312461/f9e7263c-2d38-4b1b-a2f4-9c9244b5070e)


## Contact

![https://user-images.githubusercontent.com/108312461/212851640-3e52332d-5346-4c1a-ab32-e337854afe71.png](https://user-images.githubusercontent.com/108312461/212851640-3e52332d-5346-4c1a-ab32-e337854afe71.png)


Page: [https://mi2rl.co](https://mi2rl.co/)

Email: kiduk.amc@gmail.com


## Reference
```
@Article{khosla2020supervised,
    title   = {Supervised Contrastive Learning},
    author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
    journal = {arXiv preprint arXiv:2004.11362},
    year    = {2020},
}
```
