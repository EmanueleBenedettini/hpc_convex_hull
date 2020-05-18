## This is a gnuplot script that draws a set of points and its convex
## hull as computed by the convex-hull or qhull programs.
##
## Usage:
##
## gnuplot -c plot-hull.gp input.txt output.txt hull.png
##
## where input.txt and output.txt must be replaced by the input and output
## used by convex-hull, abd hull.png is the name of the output image.
##
## Last modified on 2019-11-16 by Moreno Marzolla <moreno.marzolla(at)unibo.it

set terminal png size 1024,768 linewidth 1 notransparent fontscale 1.4
set output ARG3
set title ARG1
#unset border
#unset tics
set key off
plot ARG1 using 1:2 with points pt 5 ps .5, ARG2 using 1:2 with lines lw 3
