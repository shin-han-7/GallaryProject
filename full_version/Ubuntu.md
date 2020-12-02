## Ubuntu envs setting
01. open ubuntu
```
cmd >bash
```
```
/mnt/c/Users/ASUS$ sudo apt-get update
/mnt/c/Users/ASUS$ sudo apt-get upgrade
```
02. pip & python
```
--version
sudo apt-get install python-pip
pip -V 
sudo apt-get install python3-pip
```
03. envs
建立：mkvirtualenv xxxx<br>
刪除：revirtuallenv xxxx<br>
進入：workon xxxx<br>
退出：deactivate<br>
xxx : envs name
```
--version
pip install virtualenv
```
python3<br>
```
#1
$ sudo apt-get install python3-pip
$ pip3 install virtualenv
#2
$ which python3
$ virtualenv -p <python路徑> <想創建的環境名稱>
#3
$ source <環境名稱>/bin/activate
#4
(<環境名稱>)$
(<環境名稱>)$ deactivate
#5
pip install 安裝虛擬環境內所需套件
```
進入資料夾 : xxxx/bin$ source activate

04. pytorch_gpu envs
pytorch_gpu setting tutorial:
https://codingnote.cc/zh-tw/p/177346/
Ubuntu18.04下安装深度学习框架Pytorch（GPU加速）:
https://blog.csdn.net/wuzhiwuweisun/article/details/82753403


