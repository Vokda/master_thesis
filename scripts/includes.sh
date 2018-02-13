#!/bin/bash

#executable
executable=./skepu_ann

#data directories
data_dir=data

#solver directories
networks_dir=networks

#text formating
bold=$(tput bold)
normal=$(tput sgr0)

#this function will not do more than draw a line of '=' signs
function draw_line
{
	for i in `seq 1 80`
	do
		printf =
	done
	echo =
}

#will generate a file name of a solver.prototxt
# of the format iterations_backend_i.log
#args
#1 network directory 
#2 backend
function file_name
{
	net_dir=$1
	backend=$2
	debug=$3
	#find and echo solver
	solver=$(find $net_dir -name "solver.prototxt")
	if [ -z "$solver" ]; then #if no solver is found abort
		>&2 echo "ERROR: solver.prototxt not found! in $netdir"
		exit 1
	fi

	#if no backend is provided pick the one from the solver.txt
	if [ -z  "$backend" ]; 
	then
		backend =$(grep solver_mode $solver | egrep -oi '(cpu|gpu)' )
	fi
	#find number of test iterations in solver
	iterations=$(grep max_iter $solver | egrep -o '[0-9]+' )

	#if nr of iterations not found exit
	if [ -z "$iterations" ]; 
	then
		>&2 echo "ERROR: Number of iterations not found in ${solver}."
		exit 1
	fi

	#net_name=$(dirname $(basename $net_dir))

	i=1
	file_name=${iterations}_${backend}
	file_name_temp=${file_name}_$i
	output=$net_dir/data/$file_name_temp
	suffix=.log

	#if debug is set old name will be overwritten
	if [[ ! -z "$debug" ]] ; then
		output=$output$suffix
		echo $output
		exit 0
	fi

	output_temp=$output$suffix
	#if output already exists increase the number added to it
	while [ -f $output_temp ] 
	do
		(( i++ ))
		file_name_temp=${file_name}_$i
		output=$net_dir/data/$file_name_temp
		output_temp=$output$suffix
	done

	output=$output$suffix
	echo $output
}
