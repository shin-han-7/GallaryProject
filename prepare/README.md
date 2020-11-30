## AFADcount_local.py
update at 201130
<br>
data analyzing for AFAD

- Data size: 165501
- Data form: 15\111\109008-0.jpg
- Data range: min= 15 ,max= 72
- Age num:57 (0-57?)
- Data Frame:
```
  age gender  genID          file                 path  ageID
0  15   male      0  109008-0.jpg  15\111\109008-0.jpg      0
1  15   male      0  114557-0.jpg  15\111\114557-0.jpg      0
2  15   male      0  116002-0.jpg  15\111\116002-0.jpg      0
3  15   male      0  116080-1.jpg  15\111\116080-1.jpg      0
4  15   male      0  116596-2.jpg  15\111\116596-2.jpg      0 
```
```
attri     type
age       <object>
gender    <object>
genID     <int64>
file      <object>
path      <object>
ageID     <int32>
dtype: object
```

## AFADcount.py
update at 201130
same with local version
```
Python AFADcount.py --rootDir AFAD-Full --maleFileName 111
```
help
```
>Python AFADcount.py -h
usage: AFADcount.py [-h] --rootDir ROOTDIR --maleFileName MALEFILENAME
                    [--traincsv TRAINCSV] [--testcsv TESTCSV] [--info INFO]

optional arguments:
  -h, --help            show this help message and exit
  --rootDir ROOTDIR     get dataset dirRoot(path)
  --maleFileName MALEFILENAME
                        age/gen dataset gen file name
  --traincsv TRAINCSV   train csv file rename
  --testcsv TESTCSV     test csv file rename
  --info INFO           dataset info csv file rename
```

## Prepare Data
walk root dir
>AFAD-Full
>
>>15
>>
>>>111
>>>
>>>>./15/111/109008-0.jpg
>>>>./15/111/114557-0.jpg
>>>
>>>112
>>>
>>16
>>>111
>>>112
>>...
>>72
>>>111
>>>112
>>>
>
