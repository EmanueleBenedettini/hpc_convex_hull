/****************************************************************************
 *
 * convex-hull.c
 *
 * Compute the convex hull of a set of points in 2D
 *
 * Copyright (C) 2019 Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last updated on 2019-11-25
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************
 *
 * Questo programma calcola l'inviluppo convesso (convex hull) di un
 * insieme di punti 2D letti da standard input usando l'algoritmo
 * "gift wrapping". Le coordinate dei vertici dell'inviluppo sono
 * stampate su standard output.  Per una descrizione completa del
 * problema si veda la specifica del progetto sul sito del corso:
 *
 * http://moreno.marzolla.name/teaching/HPC/
 *
 * Per compilare:
 *
 * gcc -D_XOPEN_SOURCE=600 -std=c99 -Wall -Wpedantic -O2 convex-hull.c -o convex-hull -lm
 *
 * (il flag -D_XOPEN_SOURCE=600 e' superfluo perche' viene settato
 * nell'header "hpc.h", ma definirlo tramite la riga di comando fa si'
 * che il programma compili correttamente anche se non si include
 * "hpc.h", o per errore non lo si include come primo file).
 *
 * Per eseguire il programma si puo' usare la riga di comando:
 *
 * ./convex-hull < ace.in > ace.hull
 * 
 * Per visualizzare graficamente i punti e l'inviluppo calcolato è
 * possibile usare lo script di gnuplot (http://www.gnuplot.info/)
 * incluso nella specifica del progetto:
 *
 * gnuplot -c plot-hull.gp ace.in ace.hull ace.png
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BLKDIM 1024 // number of threads per block. has to be multiple of 2, make sure it remains in thread limits caused by cuda compute capability

//#define NO_CUDA_CHECK_ERROR //enable this during evaluation (removes some overhead)

/* A single point */
typedef struct {
    double x, y;
} point_t;

/* An array of n points */
typedef struct {
    int n;      /* number of points     */
    point_t *p; /* array of points      */
} points_t;


typedef struct {
	point_t p;	// point
	double a;	// angle
	int id;		// identifier
} cuda_point_t;

typedef struct {
	int n;				// number of points
	cuda_point_t *elem;	// array of pairs (point_t, angle, id)
} cuda_points_t;

enum {
    LEFT = -1,
    COLLINEAR,
    RIGHT
};

/**
 * Read input from file f, and store the set of points into the
 * structure pset.
 */
void read_input( FILE *f, points_t *pset )
{
    char buf[1024];
    int i, dim, npoints;
    point_t *points;
    
    if ( 1 != fscanf(f, "%d", &dim) ) {
        fprintf(stderr, "FATAL: can not read dimension\n");
        exit(EXIT_FAILURE);
    }
    if (dim != 2) {
        fprintf(stderr, "FATAL: This program supports dimension 2 only (got dimension %d instead)\n", dim);
        exit(EXIT_FAILURE);
    }
    if (NULL == fgets(buf, sizeof(buf), f)) { /* ignore rest of the line */
        fprintf(stderr, "FATAL: failed to read rest of first line\n");
        exit(EXIT_FAILURE);
    }
    if (1 != fscanf(f, "%d", &npoints)) {
        fprintf(stderr, "FATAL: can not read number of points\n");
        exit(EXIT_FAILURE);
    }
    assert(npoints > 2);
    points = (point_t*)malloc( npoints * sizeof(*points) );
    assert(points);
    for (i=0; i<npoints; i++) {
        if (2 != fscanf(f, "%lf %lf", &(points[i].x), &(points[i].y))) {
            fprintf(stderr, "FATAL: failed to get coordinates of point %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    pset->n = npoints;
    pset->p = points;
}

/**
 * Free the memory allocated by structure pset.
 */
void free_pointset( points_t *pset )
{
    pset->n = 0;
    free(pset->p);
    pset->p = NULL;
}

/**
 * Dump the convex hull to file f. The first line is the number of
 * dimensione (always 2); the second line is the number of vertices of
 * the hull PLUS ONE; the next (n+1) lines are the vertices of the
 * hull, in clockwise order. The first point is repeated twice, in
 * order to be able to plot the result using gnuplot as a closed
 * polygon
 */
void write_hull( FILE *f, const points_t *hull )
{
    int i;
    fprintf(f, "%d\n%d\n", 2, hull->n + 1);
    for (i=0; i<hull->n; i++) {
        fprintf(f, "%f %f\n", hull->p[i].x, hull->p[i].y);
    }
    /* write again the coordinates of the first point */
    fprintf(f, "%f %f\n", hull->p[0].x, hull->p[0].y);    
}

void print_debug_statements(const char* str ){
#ifdef DEBUG
    fprintf(stderr, "%s", str);
#endif
}

/**
 * Return LEFT, RIGHT or COLLINEAR depending on the shape
 * of the vectors p0p1 and p1p2
 *
 * LEFT            RIGHT           COLLINEAR
 * 
 *  p2              p1----p2            p2
 *    \            /                   /
 *     \          /                   /
 *      p1       p0                  p1
 *     /                            /
 *    /                            /
 *  p0                            p0
 *
 * See Cormen, Leiserson, Rivest and Stein, "Introduction to Algorithms",
 * 3rd ed., MIT Press, 2009, Section 33.1 "Line-Segment properties"
 */
int turn(const point_t p0, const point_t p1, const point_t p2)
{
    /*
      This function returns the correct result (COLLINEAR) also in the
      following cases:
      
      - p0==p1==p2
      - p0==p1
      - p1==p2
    */
    const double cross = (p1.x-p0.x)*(p2.y-p0.y) - (p2.x-p0.x)*(p1.y-p0.y);
    if (cross > 0.0) {
        return LEFT;
    } else {
        if (cross < 0.0) {
            return RIGHT;
        } else {
            return COLLINEAR;
        }
    }
}

/**
 * Get the clockwise angle between the line p0p1 and the vector p1p2 
 *
 *         .
 *        . 
 *       .--+ (this angle) 
 *      .   |    
 *     .    V
 *    p1--------------p2
 *    /
 *   /
 *  /
 * p0
 *
 * The function is not used in this program, but it might be useful.
 */
__host__ __device__ double cw_angle(const point_t p0, const point_t p1, const point_t p2)
{
    const double x1 = p2.x - p1.x;
    const double y1 = p2.y - p1.y;    
    const double x2 = p1.x - p0.x;
    const double y2 = p1.y - p0.y;
    const double dot = x1*x2 + y1*y2;
    const double det = x1*y2 - y1*x2;
    const double result = atan2(det, dot);
    return (result >= 0 ? result : 2*M_PI + result);
}


__device__ bool are_equals_p(const point_t p1, const point_t p2){
    if ((p1.x == p2.x) && (p1.y == p2.y)){
        return true;
    } else {
        return false;
    }
}

/**
 *	Compute the angle between p1, p2 and p3 and generate cuda_point_p data
 */
__device__ cuda_point_t compute_angle( const point_t p1, const point_t p2, const point_t p3, unsigned int p3_Id) {
    cuda_point_t p;
    p.p = p3;
    p.id = p3_Id;

    //if(p2.x != p3.x || p2.y != p3.y){
    if(are_equals_p(p1, p3) || are_equals_p(p2, p3)){
        p.a = 2*M_PI+1;
    }else{
        p.a = cw_angle(p1, p2, p3);
    }
    return p;
}

/**
 *	Return the minimum angle between the two provided (custom datatype)
 */
__device__ cuda_point_t minAngle(cuda_point_t p1, cuda_point_t p2) {
	if (p1.a < p2.a) {
		return p1;
	} else {
		return p2;
	}
}

//------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------

/*
 * This kernel is used to populate the data structure. Should be called whit a total thread count equals (or greater) than pset->n
 *
 __global__ void fill_data_structure(const points_t *pset, cuda_points_t *data_structure)
 {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < pset->n){
        data_structure->elem[i].p = pset->p[i];
        data_structure->elem[i].id = i;
    }
    if (i == 0){
        data_structure->n = pset->n;
    }
 }*/

/*
 * This kernel is used to retrieve the result in a simple way (removing addressing problems).
 *
 __global__ void ret_result( int *res, cuda_points_t *data_structure)
 {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i == 0){
        *res = data_structure->elem[0].id;
    }
 }*/

/**
 *	Reduction functions. The source is an official sheet from nvidia developer site: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 *	It has been modified to accomplish our goal
 */

//final reduction, running only on one block
/*
template <unsigned int blockSize>
__device__ void warpReduce(volatile cuda_point_t sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] = minAngle(sdata[tid], sdata[tid + 32]);
	if (blockSize >= 32) sdata[tid] = minAngle(sdata[tid], sdata[tid + 16]);
	if (blockSize >= 16) sdata[tid] = minAngle(sdata[tid], sdata[tid + 8]);
	if (blockSize >= 8) sdata[tid] = minAngle(sdata[tid], sdata[tid + 4]);
	if (blockSize >= 4) sdata[tid] = minAngle(sdata[tid], sdata[tid + 2]);
	if (blockSize >= 2) sdata[tid] = minAngle(sdata[tid], sdata[tid + 1]);
}

//main reduction
template <unsigned int blockSize>
__global__ void reduce(cuda_points_t* g_idata, cuda_points_t* g_odata, const int n, point_t prev, point_t cur) 
{
	extern __shared__ cuda_point_t sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	
	//each thread executes more than one angle operation
	while (i < n) {
		//generate point angle data and save them on shared memory
		sdata[tid] = minAngle(compute_angle(prev, cur, g_idata->elem[i].p, i), compute_angle(prev, cur, g_idata->elem[i + blockSize].p, i+blockSize));

		i += gridSize;
		__syncthreads();
	}

	//for each following lines, we have a reduction running whit log(n) threads
	if ((blockSize >= 1024) && (tid < 512)) { sdata[tid] = minAngle(sdata[tid], sdata[tid + 512]);	__syncthreads(); } 
	if ((blockSize >= 512) && (tid < 256))  { sdata[tid] = minAngle(sdata[tid], sdata[tid + 256]);	__syncthreads(); } 
	if ((blockSize >= 256) && (tid < 128))  { sdata[tid] = minAngle(sdata[tid], sdata[tid + 128]);	__syncthreads(); } 
	if ((blockSize >= 128) && (tid < 64))   { sdata[tid] = minAngle(sdata[tid], sdata[tid + 64]);	__syncthreads(); } 

    if (tid < 32){ //warpReduce(sdata, tid);
        if (blockSize >= 64) sdata[tid] = minAngle(sdata[tid], sdata[tid + 32]);
	    if (blockSize >= 32) sdata[tid] = minAngle(sdata[tid], sdata[tid + 16]);
	    if (blockSize >= 16) sdata[tid] = minAngle(sdata[tid], sdata[tid + 8]);
	    if (blockSize >= 8) sdata[tid] = minAngle(sdata[tid], sdata[tid + 4]);
	    if (blockSize >= 4) sdata[tid] = minAngle(sdata[tid], sdata[tid + 2]);
	    if (blockSize >= 2) sdata[tid] = minAngle(sdata[tid], sdata[tid + 1]);
    }
	if(tid == 0) g_odata->elem[blockIdx.x] = sdata[0];	//array of result (each block return his best result)
}*/

/**
 *  - used to reduce the elements assigned to one block
 */
__device__ cuda_point_t block_reduce(cuda_point_t *sdata, cuda_point_t *out, const unsigned int tid, const unsigned int blockSize){
    //for each following lines, we have a reduction running whit log(n) threads
	if ((blockSize >= 1024) && (tid < 512)) { sdata[tid] = minAngle(sdata[tid], sdata[tid + 512]);  __syncthreads(); }
	if ((blockSize >= 512) && (tid < 256))  { sdata[tid] = minAngle(sdata[tid], sdata[tid + 256]);  __syncthreads(); }
	if ((blockSize >= 256) && (tid < 128))  { sdata[tid] = minAngle(sdata[tid], sdata[tid + 128]);  __syncthreads(); }
	if ((blockSize >= 128) && (tid < 64))   { sdata[tid] = minAngle(sdata[tid], sdata[tid + 64]);   __syncthreads(); }
    if ((blockSize >= 64) && (tid < 32)) { sdata[tid] = minAngle(sdata[tid], sdata[tid + 32]);  __syncthreads(); }
	if ((blockSize >= 32) && (tid < 16)) { sdata[tid] = minAngle(sdata[tid], sdata[tid + 16]);  __syncthreads(); }
	if ((blockSize >= 16) && (tid < 8)) { sdata[tid] = minAngle(sdata[tid], sdata[tid + 8]);    __syncthreads(); }
	if ((blockSize >= 8) && (tid < 4)) { sdata[tid] = minAngle(sdata[tid], sdata[tid + 4]);     __syncthreads(); }
	if ((blockSize >= 4) && (tid < 2)) { sdata[tid] = minAngle(sdata[tid], sdata[tid + 2]);     __syncthreads(); }
    if ((blockSize >= 2) && (tid < 1)) { sdata[tid] = minAngle(sdata[tid], sdata[tid + 1]);     __syncthreads(); }
    

    /*
    unsigned int b_limit = BLKDIM;

    //each cycle is a reduction step. it is completed in log2(BLKDIM), so in this case whit BLKDIM = 1024  --> the cycle runs 10 times
    while (b_limit >= 2){
        if ((blockSize >= b_limit) && (tid < b_limit/2)) { 
            sdata[tid] = minAngle(sdata[tid], sdata[tid + b_limit/2]);
            //printf( "b_limit= %d reduction phase between %d & %d: id %d has found a %f angle\n", b_limit, tid, tid+b_limit/2, tid, sdata[tid].a);
            __syncthreads();
        }
        b_limit /= 2;
    }
    */

    /*if(tid == 0){ //array of result (each block return his best result)
        *out = sdata[0];	
    }*/
    return sdata[0];
}

__global__ void reduce(const points_t *pset, cuda_point_t *temp, int *result, point_t prev, point_t cur){    //temp must have #blocks lenght
    const unsigned int tid = threadIdx.x;
    const unsigned int i = threadIdx.x + blockIdx.x * BLKDIM;
    __shared__ cuda_point_t sdata[BLKDIM];

    assert(gridDim.x < BLKDIM);

    if(i<BLKDIM){
        temp[tid].a = 2*M_PI+1;
        temp[tid].id = -1;
    }
    sdata[tid].a = 2*M_PI+1;
    sdata[tid].id = -1;
    __syncthreads();

    if(tid==0) printf( "test prova prova kernel\n" );  //this if for debug ####################################

    if(i < pset->n){
        sdata[tid] = compute_angle(prev, cur, pset->p[i], i);
        __syncthreads();
        //if(1/*tid==0*/) printf( "initial data: id %d has found a %f angle\n", i, sdata[tid].a);  //this if for debug ####################################

        //block_reduce(sdata, &temp[blockIdx.x], tid, BLKDIM);
        cuda_point_t res = block_reduce(sdata, &temp[blockIdx.x], tid, BLKDIM);
        __syncthreads();

        if(tid==0) {
            printf( "reduce result=  id %d   angle %f   block %d\n", res.id, res.a, blockIdx.x);   //this if for debug ####################################
            temp[blockIdx.x].a = res.a;
            temp[blockIdx.x].id = res.id;
        }
        __syncthreads();

        if(1/*tid==0*/) printf( "1st phase: id %d has found a %f angle - %d\n", i, temp[blockIdx.x].a, blockIdx.x);   //this if for debug ####################################


        //if we have more than BLKDIM blocks allocated --- Currently limited by hardcap before kernel call
        /*if(gridDim.x > BLKDIM && i < gridDim.x ){
            sdata[tid].a = 2*M_PI+1;
            sdata[tid].id = -1;
            if(i < gridDim.x) sdata[tid] = temp[i];
            __syncthreads();

            block_reduce(sdata, &temp[blockIdx.x], tid, BLKDIM); //gridDim.x 
            __syncthreads();
        }*/
        
        
        //if we have more than 1 block allocated
        /*
        if(i < BLKDIM && gridDim.x > 1){
            sdata[tid].a = 2*M_PI+1;
            sdata[tid].id = -1;
            if(tid < gridDim.x) sdata[tid] = temp[tid];
            __syncthreads();

            block_reduce(sdata, &temp[0], tid, BLKDIM); //gridDim.x//BLKDIM  
            __syncthreads();

            //if(tid==0) printf( "2nd phase: id %d has found a %f angle\n", i, sdata[tid].a); //this if for debug ####################################
        }*/

        //this extracts the result
        if(i == 0){
            *result = temp[0].id;
            if(i==0) printf( "result is:  id= %d  angle= %f\n", temp[0].id, temp[0].a); //this if for debug ####################################
        }        
    }
}

//------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------

/**
 * Compute the convex hull of all points in pset using the "Gift
 * Wrapping" algorithm. The vertices are stored in the hull data
 * structure, that does not need to be initialized by the caller.
 */
void convex_hull(const points_t *pset, points_t *hull)
{
    const int n = pset->n;
    const point_t *p = pset->p;
    int i;
    int cur, next, leftmost;

	int blocks = (n / BLKDIM) + 1;
    fprintf(stderr, "n° blocchi = %d\n", blocks);

    cuda_point_t *temp_elem;
    points_t *d_pset;
    point_t prev_p, cur_p, fakePoint, *temp_set;
    int *d_result;

    hull->n = 0;
    /* There can be at most n points in the convex hull. At the end of
       this function we trim the excess space. */
    hull->p = (point_t*)malloc(n * sizeof(*(hull->p))); 
    assert(hull->p);
    
    /* Identify the leftmost point p[leftmost] */
    leftmost = 0;
    for (i = 1; i<n; i++) {
        if (p[i].x < p[leftmost].x) {
            leftmost = i;
        }
    }
    cur = leftmost;
    fakePoint.x = p[leftmost].x;		//
    fakePoint.y = p[leftmost].y - 1;	// This point is used for the first reduction operation that we will encounter

    cudaSafeCall(   cudaMalloc( (void **)&d_pset,   sizeof(points_t))   );
    cudaSafeCall(   cudaMalloc( (void **)&temp_elem,    blocks * sizeof(cuda_point_t))  );
    cudaSafeCall(   cudaMalloc( (void **)&temp_set,   n * sizeof(point_t))  );
    cudaSafeCall(   cudaMalloc( (void **)&d_result, sizeof(int))  );

    cudaSafeCall(   cudaMemcpy(temp_set, pset->p, n * sizeof(point_t), cudaMemcpyHostToDevice)  );
    cudaSafeCall(   cudaMemcpy(&d_pset->p, &temp_set, sizeof(point_t*), cudaMemcpyHostToDevice)   );
    cudaSafeCall(   cudaMemcpy(&d_pset->n, &pset->n, sizeof(int), cudaMemcpyHostToDevice)   );

    cudaCheckError();
    
    /* Main loop of the Gift Wrapping algorithm. This is where most of
       the time is spent; therefore, this is the block of code that
       must be parallelized. */
    do {
        /* Add the current vertex to the hull */
        assert(hull->n < n);
        hull->p[hull->n] = p[cur];
        hull->n++;

        fprintf(stderr, "n. punti trovati: %d\n", hull->n);

        if(hull->n > 1){
            prev_p = hull->p[hull->n-2];
        }else{
            prev_p = fakePoint; //used only for the first iteration
        }

        cur_p = hull->p[hull->n-1];
        
        assert(blocks < 1024);  //hard cap, could be a limitation
 
        reduce<<<blocks, BLKDIM >>>(d_pset, temp_elem, d_result, prev_p, cur_p);    //temp_elem must have #blocks elements
        cudaDeviceSynchronize();
        cudaCheckError();

        cudaSafeCall(   cudaMemcpy(&next, d_result, sizeof(int), cudaMemcpyDeviceToHost)    );
        cudaCheckError();

        if(hull->n >= n)break;
        break;
        //fprintf(stderr, "next= %d\n", next);

        assert(next >= 0);  //if this is not true, the reduction is not working
        assert(cur != next);
        cur = next;
    } while (cur != leftmost);

    //free of gpu memory allocation
    //print_debug_statements("debug point 3.1 - deallocation of memory\n");
    cudaFree(d_result);
    cudaFree(temp_set);
    cudaFree(temp_elem);
    cudaFree(d_pset);
    
    /* Trim the excess space in the convex hull array */
    hull->p = (point_t*)realloc(hull->p, (hull->n) * sizeof(*(hull->p)));
    assert(hull->p); 
}

/**
 * Compute the area ("volume", in qconvex terminology) of a convex
 * polygon whose vertices are stored in pset using Gauss' area formula
 * (also known as the "shoelace formula"). See:
 *
 * https://en.wikipedia.org/wiki/Shoelace_formula
 *
 * This function does not need to be parallelized.
 */
double hull_volume( const points_t *hull )
{
    const int n = hull->n;
    const point_t *p = hull->p;
    double sum = 0.0;
    int i;
    for (i=0; i<n-1; i++) {
        sum += ( p[i].x * p[i+1].y - p[i+1].x * p[i].y );
    }
    sum += p[n-1].x*p[0].y - p[0].x*p[n-1].y;
    return 0.5*fabs(sum);
}

/**
 * Compute the length of the perimeter ("facet area", in qconvex
 * terminoloty) of a convex polygon whose vertices are stored in pset.
 * This function does not need to be parallelized.
 */
double hull_facet_area( const points_t *hull )
{
    const int n = hull->n;
    const point_t *p = hull->p;
    double length = 0.0;
    int i;
    for (i=0; i<n-1; i++) {
        length += hypot( p[i].x - p[i+1].x, p[i].y - p[i+1].y );
    }
    /* Add the n-th side connecting point n-1 to point 0 */
    length += hypot( p[n-1].x - p[0].x, p[n-1].y - p[0].y );
    return length;
}

int main( void )
{
    points_t pset, hull;
    double tstart, elapsed;
    
    read_input(stdin, &pset);
    tstart = hpc_gettime();
    convex_hull(&pset, &hull);
    elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "\nConvex hull of %d points in 2-d:\n\n", pset.n);
    fprintf(stderr, "  Number of vertices: %d\n", hull.n);
    fprintf(stderr, "  Total facet area: %f\n", hull_facet_area(&hull));
    fprintf(stderr, "  Total volume: %f\n\n", hull_volume(&hull));
    fprintf(stderr, "Elapsed time: %f\n\n", elapsed);
    write_hull(stdout, &hull);
    free_pointset(&pset);
    free_pointset(&hull);
    return EXIT_SUCCESS;    
}
