# GallaryProject
### STEP01.AFAD_Dataset
- 移動至`./dataset/tarball/`
- 執行```$ sh restore.sh```, get`AFAD-Lite.tar.xz `
- 安裝套件```$ sudo apt-get install -y xz-utils```
- 解壓縮```$ tar Jxvf AFAD-Lite.tar.xz```, get`/AFAD-Full`

### STEP02.Prepare_Data_to_cvs
- ```> Python AFADcount_local.py```
- get`training_set.cvs`...

### STEP03.Training
- ```> Python PAFAD_train_age.py -h```
- ```> Python AFAD_gen_train.py -h```

### STEP04.Demo
