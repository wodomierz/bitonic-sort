#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;
int* bitonic_sort(int* to_sort, int n){
    cuInit(0);
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot acquire device 0\n");
        exit(1);
    }
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        printf("cannot create Kontext\n");
        exit(1);
    }

    CUmodule cuModule = (CUmodule)0;
    res = cuModuleLoad(&cuModule, "bitonic_sort.ptx");
    if (res != CUDA_SUCCESS) {
        printf("cannot load module: %d\n", res);
        exit(1);
    }

    CUfunction bitonic_sort;
    res = cuModuleGetFunction(&bitonic_sort, cuModule, "bitonic_sort");
    CUfunction bitonic_merge;
    res = cuModuleGetFunction(&bitonic_merge, cuModule, "bitonic_sort");
    CUfunction bitonic_triangle_merge;
    res = cuModuleGetFunction(&bitonic_triangle_merge, cuModule, "bitonic_sort");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    int numberOfBlocks = (n+1023)/1024;

    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, n*sizeof(int));

    cuMemcpyHtoD(deviceToSort, to_sort, n * sizeof(int));

    int local_n = 2*1024;
    void* args[2] =  { &deviceToSort, &local_n};
    cuLaunchKernel(bitonic_sort, numberOfBlocks, 1, 1, 1024, 1, 1, 0, 0, args, 0);
    cuCtxSynchronize();

    for (int d_traingle = local_n*2; d_traingle <= n; d_traingle*=2) {
        printf("hello\n");
        void* args1[3] = { &deviceToSort, &d_traingle, &n};
        cuLaunchKernel(bitonic_triangle_merge, numberOfBlocks, 1, 1, 1024, 1, 1, 0, 0, args1, 0);
        cuCtxSynchronize();
        for (int d = d_traingle/2; d >= 2; d /= 2) {
            void* args2[3] = { &deviceToSort, &d, &n};
            cuLaunchKernel(bitonic_merge, numberOfBlocks, 1, 1, 1024, 1, 1, 0, 0, args2, 0);
            cuCtxSynchronize();
        }
    }

    
        
    int* result = (int*) malloc(sizeof(int) * n);

    cuMemcpyDtoH((void*)result, deviceToSort, n * sizeof(int));

    cuMemFree(deviceToSort);
    cuCtxDestroy(cuContext);
    return result;
}
