#!/bin/bash

dir=backup_`date +%y%m%d`

mkdir backup/$dir

cp *.cpp backup/$dir 
cp *.hpp backup/$dir
cp Make* backup/$dir
