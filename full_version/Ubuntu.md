## Ubuntu envs setting
01. open cmd
```
cmd >bash
```
```
/mnt/c/Users/ASUS$ sudo apt-get update
/mnt/c/Users/ASUS$ sudo apt-get upgrade
```
02. pip & python
```
pip --version
sudo apt-get install python-pip
pip3 -V 
sudo apt-get install python3-pip
```
03. envs <br>
python3
```
#1
$ virtualenv --version
$ pip3 install virtualenv
$ sudo apt-get install python3-venv
#2
path to target evn dir
$ python3 -m venv pytorch_gpu
#3
$ source pytorch_gpu/bin/activate
#4
(<環境名稱>)$
#5
$ pip install 安裝虛擬環境內所需套件
#6
(<環境名稱>)$ deactivate
```

04. pytorch_gpu envs
pytorch_gpu setting tutorial:<br>
https://codingnote.cc/zh-tw/p/177346/ <br>
Ubuntu18.04下安装深度学习框架Pytorch（GPU加速）:<br>
https://blog.csdn.net/wuzhiwuweisun/article/details/82753403 <br>

05. Other reference
liunx ins: <br>
https://blog.techbridge.cc/2017/12/23/linux-commnd-line-tutorial/



