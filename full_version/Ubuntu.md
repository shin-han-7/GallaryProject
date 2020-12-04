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
#1 setting
$ virtualenv --version
$ pip3 install virtualenv
$ sudo apt-get install python3-venv

#2 create
path to target evn dir
$ python3 -m venv pytorch_gpu

#3 activate env
$ source pytorch_gpu/bin/activate
(<環境名稱>)$

#4 pip
(<環境名稱>)$ pip install 安裝虛擬環境內所需套件

#5 log out
(<環境名稱>)$ deactivate

#6 delete env
$ rm -rf venv
```
04. pytorch_gpu install
Pytorch office <br>
https://pytorch.org/ <br>
Pytorch gpu verssion install <br>
https://mark-down-now.blogspot.com/2018/05/pytorch-gpu-ubuntu-1804.html <br>

05. Other reference
liunx ins: <br>
https://blog.techbridge.cc/2017/12/23/linux-commnd-line-tutorial/ <br>
Download for Linux and Unix:<br>
https://git-scm.com/download/linux



