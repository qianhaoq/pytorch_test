#!/bin/sh
# 从图片全库中提取指定数量的图片
# examples:
#    sh pickup.sh 20  (提取前20张图片)
# author qh

set -x

number=$1
echo ${number}
# 用户主目录
home_path=`cd ../../../ && pwd`
# 原始数据地址
data_path="${home_path}/data/origin/"
# 提取数据地址
pickup_path="${home_path}/data/train/"

# 清除上次pickup文件
rm -r ${pickup_path}/*

for i in `seq -f "%03g" 802`
do
    cd ${data_path}/${i}
    mkdir ${pickup_path}/${i}
    j=0
    for j in `ls`
    do
        id=`echo ${j} |awk -F '.' '{print $1}'`
        if [ $id -lt 10 ]:
        then cp ${data_path}/${i}/${j} ${pickup_path}/${i}/${j}
        fi
        #folder_list[j]="$i"
        #j=`expr $j + 1`
    done
done