#!/bin/bash

for f in *.in
do
	inputName=$f
	hullName=${f:0:-3}.hull
	pngName=${f:0:-3}.png

	echo "processing $f _________________________________________________________________________________"
	#echo $hullName
	#echo $pngName
	
	./cuda-convex-hull < $inputName > $hullName
	gnuplot -c plot-hull.gp $inputName $hullName $pngName
done