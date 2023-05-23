# SimChest: a novel anomaly agnostic model for similarity measurement in follow-up chest radiograph pairs via a supervised contrastive learning model

This is a PyTorch implementation of the LDH under reivew paper:

## Pretraining task procedure

![스크린샷 2023-05-23 오후 12 40 56](https://github.com/chokyungjin/Bi_rads/assets/108312461/aaf8e058-a71d-409c-abac-c9248fff92e6)


## Model training

```

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_supcon.py --dataset real --name SupCon_scratch \
--print_freq=5 --save_freq 1 --num_workers 8 --aug True --warm \
--batch_size 8 --model resnet50 --method SupCon --epochs 100

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

Email: [kjcho](mailto:kjcho@amc.seoul.kr).amc@gmail.com


## Reference
```
@Article{khosla2020supervised,
    title   = {Supervised Contrastive Learning},
    author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
    journal = {arXiv preprint arXiv:2004.11362},
    year    = {2020},
}
```