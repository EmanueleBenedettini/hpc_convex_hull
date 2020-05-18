#!/bin/bash

#Written by Emanuele Benedettini

for y in 1 2 3 #execute the tests 3 times
do
	for x in 1 2 3 4 5 6 7 8 9 10 #test for 1...10 threads
	do
		mkdir -p openMP_Results/Run_${y}/${x}_threads
		for f in *.in
		do
			inputName=$f
			hullName=openMP_Results/Run_${y}/${x}_threads/${f:0:-3}.hull
			pngName=openMP_Results/Run_${y}/${x}_threads/${f:0:-3}.png
			logName=openMP_Results/Run_${y}/${x}_threads/${f:0:-3}log.txt
	
			OMP_NUM_THREADS=$x ./omp-convex-hull < $inputName > $hullName 2> $logName

			#commentare il plot se non serve
			gnuplot -c plot-hull.gp $inputName $hullName $pngName
		done
	done
done