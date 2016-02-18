#include <cstdio>



extern "C" {

__device__
void min_max(int* for_min, int* for_max) {
	int min = *for_min;
	int max = *for_max;
	if (max < min) {
		atomicExch(for_max, min);
		atomicExch(for_min, max);
	}
};


__global__
void bitonic_sort(int* in, int n) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;	
		
	int d_traingle;
	int local_thid;
	int opposite;


	for (d_traingle = 2; d_traingle <= n; d_traingle*=2) {
		local_thid = thid % d_traingle;		
		opposite = thid - local_thid + d_traingle - 1 - local_thid;
		if (local_thid < d_traingle/2) {
			min_max(in + thid, in + opposite);
		}

		__syncthreads();

		for (int d = d_traingle/2; d >= 2; d /= 2) {
			local_thid = thid % d;	
			if (local_thid < d/2) {
				opposite = thid + d/2;
				min_max(in + thid, in + opposite);
			}
			__syncthreads();
		}
		__syncthreads();
	}

}

__global__
void bitonic_merge(int* in, int d) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;	
 	int local_thid = thid % d;	
 	int opposite = thid + d/2;
	if (local_thid < d/2) {
		min_max(in + thid, in + opposite);
	}
}

__global__
void bitonic_triangle_merge(int* in, int d_traingle) {
 	int thid = blockIdx.x * blockDim.x + threadIdx.x;
 	// printf("%d %d traingle thid \n", d_traingle, thid);	
	int local_thid = thid % d_traingle;		
	int opposite = thid - local_thid + d_traingle - 1 - local_thid;
	if (local_thid < d_traingle/2) {
		min_max(in + thid, in + opposite);
	}
}



}



