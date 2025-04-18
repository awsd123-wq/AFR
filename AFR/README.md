# AFR
 

## Training   # #培训   培训

### Setup   # # #设置   设置

**Please download the extracted I3D features for XD-Violence and UCF-Crime dataset from links below:请从以下链接下载XD-Violence和UCF-Crime数据集的I3D特征：**

> [**XD-Violence I3D onedrive   XD-Violence I3D驱动XD-Violence I3D驱动**](https://cqueducn0-my.sharepoint.com/:f:/g/personal/zbqian_cqu_edu_cn/EqnWl_Nm3h1Crjnq24wusEgB04Kvabbs_8eqMKgDHXieBA?e=x89K5f)

> [**UCF-Crime I3D onedrive**](https://cqueducn0-my.sharepoint.com/:f:/g/personal/zbqian_cqu_edu_cn/EqnWl_Nm3h1Crjnq24wusEgB04Kvabbs_8eqMKgDHXieBA?e=x89K5f)> [**UCF-Crime I3D oneddrive **]（https://cqueducn0-my.sharepoint.com/:f:/g/personal/zbqian_cqu_edu_cn/EqnWl_Nm3h1Crjnq24wusEgB04Kvabbs_8eqMKgDHXieBA?e=x89K5f）

**After downloading, please put it in the `data/` folder (No need to unzip)****下载后请放入‘ data/ ’文件夹（无需解压缩）**


### Train and Test   训练和测试   培训和测试
```shell   “‘壳
# For XD-Violence   # xd暴力   # xd暴力
python train.py --zip_feats data/xdviolence_i3d_w16_s16.zip --ampPython train.py——zip_专长数据/xdviolence_i3d_w16_s16.zipPython train.py——zip_专长数据/xdviolence_i3d_w16_s16.zip


# For UCF-Crime   #为ucf犯罪   #为ucf犯罪
python train.py --zip_feats data/ucfcrime_i3d_roc_ng_w16_s16.zip --ampPython train.py——zip_专长数据/ucfcrime_i3d_roc_ng_w16_s16.zip
```


