#!/bin/bash

if [ -z $1 ];
then
	echo "How to use: $0 network_directory/"
	exit
fi

backend=""
if [ -z $2 ];
then
	echo "No backend provided. CPU will be used."
	backend="cpu"
else
	echo "Backend $2 has been provided."
	backend=$2
fi

gdb --args skepu_ann $1/solver.prototxt $backend
