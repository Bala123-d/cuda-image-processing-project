#include <stdio.h>
#include <cuda_runtime.h>

__global__ void processData(int *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        data[i] = data[i] * 2;
    }
}

int main()
{
    int n = 1024;
    int size = n * sizeof(int);

    int *h_data;
    int *d_data;

    h_data = (int*)malloc(size);

    for (int i = 0; i < n; i++)
    {
        h_data[i] = i;
    }

    cudaMalloc((void**)&d_data, size);

    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    processData<<<4,256>>>(d_data, n);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    printf("GPU Processing Completed\n");

    cudaFree(d_data);
    free(h_data);

    return 0;
}
