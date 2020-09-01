#!/bin/bash

#Written by Emanuele Benedettini

for y in 1 2 3 #execute the tests 3 times
do
	for x in 1 2 3 4 5 6 7 8 9 10 #test for 1...10 threads
	do	
		mkdir -p MPI_Results/Run_${y}/${x}_threads
		#touch MPI_Results/Run_${y}/Run_${y}-${x}_result.txt
		for f in *.in
		do
			inputName=$f
			hullName=MPI_Results/Run_${y}/${x}_threads/${f:0:-3}.hull
			pngName=MPI_Results/Run_${y}/${x}_threads/${f:0:-3}.png
			logName=MPI_Results/Run_${y}/${x}_threads/${f:0:-3}log.txt
	
			mpirun -n $x ./mpi-convex-hull $inputName > $hullName 2> $logName

			#cat MPI_Results/Run_${y}/Run_${y}-${x}_result.txt $logName > MPI_Results/Run_${y}/Run_${y}-${x}_result.txt

			#commentare il plot se non serve
			gnuplot -c plot-hull.gp $inputName $hullName $pngName
		done
	done
done