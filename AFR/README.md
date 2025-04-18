# AFR
 

## Training

### Setup

**Please download the extracted I3D features for XD-Violence and UCF-Crime dataset from links below:请从以下链接下载XD-Violence和UCF-Crime数据集的I3D特征：**

> [**XD-Violence I3D onedrive   XD-Violence I3D驱动**](https://cqueducn0-my.sharepoint.com/:f:/g/personal/zbqian_cqu_edu_cn/EqnWl_Nm3h1Crjnq24wusEgB04Kvabbs_8eqMKgDHXieBA?e=x89K5f)

> [**UCF-Crime I3D onedrive**](https://cqueducn0-my.sharepoint.com/:f:/g/personal/zbqian_cqu_edu_cn/EqnWl_Nm3h1Crjnq24wusEgB04Kvabbs_8eqMKgDHXieBA?e=x89K5f)

**After downloading, please put it in the `data/` folder (No need to unzip)**


### Train and Test
```shell
# For XD-Violence
python train.py --zip_feats data/xdviolence_i3d_w16_s16.zip --amp


# For UCF-Crime
python train.py --zip_feats data/ucfcrime_i3d_roc_ng_w16_s16.zip --amp
```

## Quote
```bibtex
@inproceedings{vad_cmsil,
  author       = {Qian, Zhangbin and Tan, Jiawei and Ou, Zhilong and Wang, Hongxing},
  title        = {CLIP-Driven Multi-Scale Instance Learning for Weakly Supervised Video Anomaly Detection},
  booktitle    = {2024 IEEE International Conference on Multimedia and Expo (ICME)}, 
  pages        = {1--6},
  year         = {2024}
}

```
