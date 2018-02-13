#!/bin/bash
# will clean all data that is in the  networks directory
# but NOT the data in the data/ directories.
# ...unless given then -A flag

source $(dirname $0)/includes.sh

if [[ -d $1 ]];
then
	rm $1/data/*
else
	for network in $(ls -D $networks_dir)
	do
		net_dir=$networks_dir/$network 
		#data=$data_dir/$dir	

		#remove old data
		rm -vf $net_dir/*.dat $net_dir/*.pdf $net_dir/*.weights

		if [ $# -eq  0 ];
		then 
			exit
		fi

		if [ "$1" == "-A" ]
		then
			echo "Cleaning all data!"
			rm -vrf $net_dir/data/*
		fi
	done
fi
