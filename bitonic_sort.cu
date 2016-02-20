#include <cstdio>



extern "C" {

__device__
static int THREADS_IN_BLOCK = 1024;

__device__
void min_max(int* tab, int for_min, int for_max, int size) {
	if (for_min >= size || for_max >= size) {
		return;
	}
	int min = tab[for_min];
	int max = tab[for_max];
	if (max < min) {
		atomicExch(tab + for_max, min);
		atomicExch(tab + for_min, max);
	}
};


__global__
void bitonic_sort(int* to_sort, int size) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	int thid = x + y*gridDim.x;

	if (thid >= size) {
		return;
	}

	int d_traingle;
	int local_thid;
	int opposite;


	for (d_traingle = 2; d_traingle <= THREADS_IN_BLOCK; d_traingle*=2) {
		local_thid = thid % d_traingle;		
		opposite = thid - local_thid + d_traingle - 1 - local_thid;
		if (local_thid < d_traingle/2) {
			min_max(to_sort, thid, opposite, size);
		}

		__syncthreads();

		for (int d = d_traingle/2; d >= 2; d /= 2) {
			local_thid = thid % d;	
			if (local_thid < d/2) {
				opposite = thid + d/2;
				min_max(to_sort, thid, opposite, size);
			}
			__syncthreads();
		}
		__syncthreads();
	}

}

__global__
void bitonic_merge(int* to_sort, int d, int size) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thid = x + y*gridDim.x*blockDim.x;


	if (thid >= size) {
		return;
	}

 	int local_thid = thid % d;	
 	int opposite = thid + d/2;
	if (local_thid < d/2) {
		min_max(to_sort, thid,  opposite, size);
	}
}

__global__
void bitonic_triangle_merge(int* to_sort, int d_traingle, int size) {
 	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thid = x + y*gridDim.x*blockDim.x;

 	if (thid >= size) {
		return;
	}

	int local_thid = thid % d_traingle;		
	int opposite = thid - local_thid + d_traingle - 1 - local_thid;
	if (local_thid < d_traingle/2) {
		min_max(to_sort, thid,  opposite, size);
	}
}



}



