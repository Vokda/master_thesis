#!/bin/bash

if (( $# < 1 ))
then
	echo usage: [train_data_file] #[test_data_file]
	exit 1
fi

train_file=$1
test_file=$2
max_val=255
min_val=0
width=3
height=3
colors=1
data_set_size=100
nr_classes=10
samples_per_class=$((data_set_size / nr_classes))

#templates
template0='0 0 0 0 0 0 0 0 1' #0
template1='0 0 0 0 0 0 0 1 1' #1
template2='0 0 0 0 0 0 1 1 0' #2
template3='0 0 0 0 0 1 1 0 1' #3
template4='0 0 0 0 1 0 1 1 0' #4
template5='0 1 1 1 0 0 0 1 1' #5
template6='0 0 1 0 0 0 0 0 0' #6
template7='0 1 0 1 1 1 1 1 1' #7
template8='1 0 1 1 0 1 0 0 0' #8
template9='0 1 0 0 0 1 0 0 1' #9

#echo $template4

#empty files
> $train_file
> $test_file

data_size=$((width * height * colors))


d2b=({0..1}{0..1}{0..1}{0..1}{0..1}{0..1}{0..1}{0..1}{0..1})

#generate data
#for each class
for i in $(seq 0 $((nr_classes-1)))
do
	for j in $(seq 0 $(($samples_per_class-1)))
	do
		t=`echo ${d2b[i]}$i | sed -e 's/./& /g'`
		echo $t >> $train_file
	done

done

shuf $train_file --output=$train_file

#make training data
line="$width $height $colors $data_set_size $nr_classes"
#echo $line >> $train_file
sed -i "1i$line" $train_file 
echo 'second cat'
cat $train_file
