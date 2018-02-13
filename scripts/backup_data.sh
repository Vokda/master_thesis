#!/bin/bash


if (( "$#" < 1 )); then
	echo "Usage: "$0" directory/network/"
	exit 0
fi

dir=backup_`date +%y%m%d`

data=${1#*/}
dir=${dir}_$network

mkdir backup/$dir

cp $1/data/* backup/$dir 
