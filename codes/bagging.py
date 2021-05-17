
import os
import csv
import pandas as pd
path = "/home/linhw/myproject/cvdl/Imagenette/csv/" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
csv = []
lst = []
for file in files: #遍历文件夹
    csv = pd.read_csv(path+file)
    lst.append(list(csv.Category))

final = []
for index in range(9160):
    vote = []
    for cate in lst:
        vote.append(cate[index])
    final.append(max(vote, key=vote.count))

dat = pd.DataFrame({'Id': ['0007'+ '0'*(4-len(str(i)))+ str(i)+'.jpg' for i in range(9160)], 'Category': final})
dat.to_csv('bagging.csv', index=False)

