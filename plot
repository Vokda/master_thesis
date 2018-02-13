#!/bin/bash

if (($# < 1))
then
    echo "How to use: $0 [network_dir]"
    exit 1
fi

network_dir="$1/data"
caffe_network_dir="caffe/$network_dir"

#files=`ls $network_dir`
data=`ls $network_dir | grep -E '[0-9]{5,}.+1\.log'`
caffe_data=`ls $caffe_network_dir | grep -E '[0-9]{5,}.+.train'`
caffe_data=$(echo $caffe_data | awk '{print $2}')
f=`readlink -f $data`
cf=`readlink -f $caffe_data`
echo $f
echo $cf
python scripts/plot.py $f $cf
