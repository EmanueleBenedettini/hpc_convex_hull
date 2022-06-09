# README #

This project refers as the High Performance Computing exam aa.2019/20

### What is this repository for? ###

* We had to parallelize a code privided by Professor Moreno Marzolla, utilizing openMP and MPI/CUDA, in 2 separate versions.

### How do I use it? ###

* This program run on Linux distribution (Tested whit Ubuntu 16.04 LTS).
* To compile execute the command "make"
* To clean the solution from previously compiled/result files, execute the command "make clean"
* To run the program execute the command "./convex-hull < your_file.in > your_file.hull". Note that ./convex-hull can be changed whit the program you want to execute.
* To visualize the result printing starting data and result data in a image, execute the command "gnuplot -c plot-hull.gp your_file.in your_file.hull your_file.png"

* I have provided some script to automate the evaluation:
* "plotter" runs that version for all the .in files
* "Test_openMP" runs that version 3 times, with 1 to 12 core for each run, and saves the result in a separate folder.
