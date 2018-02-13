#find out if file contains skepu attributes
if egrep '\[\[[a-z]+\:\:[a-z]+\]\]' $1; then #--quiet; then
	echo $1 needs to be precompiled.
	exit 1
else
	echo $1 does not need to be precompiled.
	exit 0
fi
