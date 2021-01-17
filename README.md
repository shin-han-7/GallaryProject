# GalleryProject
分析美術館單一畫框前，觀看人次計數，以及觀看者性別、年齡區段分布。

## System Flow
```
(1) get video (.mp4)
(2) object tracking (MOT)
(3) get object ID (MOT)
(4) recognize face
(5) people counting + age/gender recognize
```

## Tracking
```
(1) set `video.mp4` in file: InOut/【FILENAME】/1_input/video.mp4
(2) run `MOT/demo.py`
    - `track.py` get obj ID/ xy coordinate
(3) output `frame/`,`person/`,`result.txt` in file: InOut/【FILENAME】/2_MOT_output/
```
Reference source: https://github.com/Zhongdao/Towards-Realtime-MOT

## Recognize
### STEP01.AFAD_Dataset
- cd dataset
- 移動至`./dataset/tarball/`
- 執行```$ sh restore.sh```, get`AFAD-Lite.tar.xz `
- 安裝套件```$ sudo apt-get install -y xz-utils```
- 解壓縮```$ tar Jxvf AFAD-Lite.tar.xz```, get`/AFAD-Full`

### STEP02.Prepare_Data_to_cvs
- cd prepare
- ```> Python AFADcount_local.py```
- get`training_set.cvs`...

### STEP03.Training
- cd training
- ```> Python age_identity.py -h```
- ```> Python AFAD_gen_train.py -h```
- output model.pt/logging.log in : ???
- set model at model/

### STEP04.Demo
- cd deom
- run `demo_identity_MOT2.py`, get `../InOut/【FILENAME】/3_final_output/0_PredictSummary.csv`
- run `Summary.py`, output final in `../InOut/【FILENAME】/3_final_output/`
