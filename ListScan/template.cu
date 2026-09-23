#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>

#define BLOCK_SIZE 512 //do not change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *aux, int len) {
    //@@ Modify the body of this kernel to generate the scanned blocks
    //@@ Make sure to use the workefficient version of the parallel scan
    //@@ Also make sure to store the block sum to the aux array 
    __shared__ float XY[2 * BLOCK_SIZE];
    int tx = threadIdx.x;
    int i = (2*blockIdx.x*blockDim.x) + tx;

    XY[tx] = (i < len) ? input[i] : 0.0f;
    XY[tx + blockDim.x] = (i + blockDim.x < len) ? input[i + blockDim.x] : 0.0f;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (tx + 1) * stride * 2 - 1;
        if (index < 2 * blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tx + 1) * stride * 2 - 1;
        if (index + stride < 2 * blockDim.x) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    if (i < len) {
        output[i] = XY[tx];
    }
    if (i + blockDim.x < len) {
        output[i + blockDim.x] = XY[tx + blockDim.x];
    }
    if (aux != NULL && tx == 0) {
        aux[blockIdx.x] = XY[2 * blockDim.x - 1];
    }
}

__global__ void addScannedBlockSums(float *output, float *aux, int len) {
	//@@ Modify the body of this kernel to add scanned block sums to 
	//@@ all values of the scanned blocks
    int tx = threadIdx.x;
    int i = (2 * blockIdx.x * blockDim.x) + tx;

    if (blockIdx.x > 0) {
        float addValue = aux[blockIdx.x - 1];
        
        if (i < len) {
            output[i] += addValue;
        }
        if (i + blockDim.x < len) {
            output[i + blockDim.x] += addValue;
        }
    }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output 1D list
  float *deviceInput;
  float *deviceOutput;
  float *deviceAuxArray, *deviceAuxScannedArray;
  int numElements; // number of elements in the input/output list. 
				   
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating device memory.");
  //@@ Allocate device memory
  //you can assume that deviceAuxArray size would not need to be more than BLOCK_SIZE*2 (i.e., 1024)
  int numBlocks = (numElements + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE);
  wbCheck(cudaMalloc((void**)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceAuxArray, numBlocks * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceAuxScannedArray, numBlocks * sizeof(float)));
  wbTime_stop(GPU, "Allocating device memory.");

  wbTime_start(GPU, "Clearing output device memory.");
  //@@ zero out the deviceOutput using cudaMemset() by uncommenting the below line
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(deviceAuxArray, 0, numBlocks * sizeof(float)));
  wbCheck(cudaMemset(deviceAuxScannedArray, 0, numBlocks * sizeof(float)));
  wbTime_stop(GPU, "Clearing output device memory.");

  wbTime_start(GPU, "Copying input host memory to device.");
  //@@ Copy input host memory to device	
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input host memory to device.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimGridAux(1, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  //@@ You need to launch scan kernel twice: 1) for generating scanned blocks 
  //@@ (hint: pass deviceAuxArray to the aux parameter)
  //@@ and 2) for generating scanned aux array that has the scanned block sums. 
  //@@ (hint: pass NULL to the aux parameter)
  //@@ Then you should call addScannedBlockSums kernel.
  scan <<<dimGrid, dimBlock >>> (deviceInput, deviceOutput, deviceAuxArray, numElements);
  scan <<<dimGridAux, dimBlock >>> (deviceAuxArray, deviceAuxScannedArray, NULL, numBlocks);
  addScannedBlockSums <<<dimGrid, dimBlock >>> (deviceOutput, deviceAuxScannedArray, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output device memory to host");
  //@@ Copy results from device to host	
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output device memory to host");

  wbTime_start(GPU, "Freeing device memory");
  //@@ Deallocate device memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));
  wbCheck(cudaFree(deviceAuxArray));
  wbCheck(cudaFree(deviceAuxScannedArray));
  wbTime_stop(GPU, "Freeing device memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
