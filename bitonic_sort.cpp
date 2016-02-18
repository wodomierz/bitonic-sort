#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <climits> 

using namespace std;
int* bitonic_sort(int* to_sort, int size){
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
    if (res != CUDA_SUCCESS) printf("some error %d\n", __LINE__);

    CUfunction bitonic_merge;
    res = cuModuleGetFunction(&bitonic_merge, cuModule, "bitonic_merge");
    if (res != CUDA_SUCCESS) printf("some error %d\n", __LINE__);

    CUfunction bitonic_triangle_merge;
    res = cuModuleGetFunction(&bitonic_triangle_merge, cuModule, "bitonic_triangle_merge");
    if (res != CUDA_SUCCESS) printf("some error %d\n", __LINE__);

    int numberOfBlocks = (size+1023)/1024;

    

    int* result = (int*) malloc(sizeof(int) * size);
    cuMemHostRegister((void*) result, size*sizeof(int), 0);
    cuMemHostRegister((void*) to_sort, size*sizeof(int), 0);

    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, size*sizeof(int));
    cuMemcpyHtoD(deviceToSort, to_sort, size * sizeof(int));

    int local_n = 1024;
    void* args[3] =  { &deviceToSort, &local_n, &size};
    cuLaunchKernel(bitonic_sort, numberOfBlocks, 1, 1, 1024, 1, 1, 0, 0, args, 0);
    cuCtxSynchronize();


    int n;
    //fit n to power of 2
    for (n = 1; n<size; n<<=1);
    for (int d_traingle = local_n*2; d_traingle <= n; d_traingle*=2) {
        void* args1[3] = { &deviceToSort, &d_traingle, &size};

        res = cuLaunchKernel(bitonic_triangle_merge, numberOfBlocks, 1, 1, 1024, 1, 1, 0, 0, args1, 0);
        if (res != CUDA_SUCCESS) printf("some error %d\n", __LINE__);
        cuCtxSynchronize();

        for (int d = d_traingle/2; d >= 2; d /= 2) {
            void* args2[3] = { &deviceToSort, &d, &size};

            res = cuLaunchKernel(bitonic_merge, numberOfBlocks, 1, 1, 1024, 1, 1, 0, 0, args2, 0);
            if (res != CUDA_SUCCESS) printf("some error %d\n", __LINE__);
            cuCtxSynchronize();
        }
    }

    cuCtxSynchronize();
    
        
    

    cuMemcpyDtoH((void*)result, deviceToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(result);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
    return result;
}
