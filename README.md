# CUDA GPU Parallel Programming Projects

A collection of GPU parallel programming projects implemented in CUDA C/C++. These projects were completed while studying parallel programming and GPU computing and explore how common algorithms can be mapped to CUDA's thread, block, and grid execution model.

The repository includes implementations of matrix multiplication, convolution, histogram computation, parallel scan, and vector addition.

Each directory contains the CUDA source code for an individual parallel-programming problem.

## Projects

**Vector Addition**             | Parallel vector addition using CUDA kernels, with work distributed across GPU threads.

**Basic Matrix Multiplication** | GPU implementation of matrix multiplication using CUDA's multidimensional thread and block organization.    

**Tiled Matrix Multiplication** | Matrix multiplication implemented using a tiled approach to explore GPU memory access and more efficient parallel execution. 

**Convolution**                 | Parallel convolution implementation demonstrating how neighboring input elements can be processed concurrently on the GPU. 

**Histogram**                   | GPU implementation of histogram computation, a problem involving many threads updating a limited number of output values. 

**List Scan**                   | Parallel prefix-scan implementation demonstrating a common building block used in many GPU algorithms.                       

## Concepts Explored

These projects provided hands-on experience with:

* Writing and launching CUDA kernels
* Organizing work using threads, blocks, and grids
* Transferring data between CPU and GPU memory
* Mapping sequential algorithms to parallel implementations
* Synchronization between GPU threads
* GPU memory access patterns
* Parallel algorithm design
* Working with multidimensional data on the GPU

## Building and Running

These programs require:

* An NVIDIA CUDA-capable GPU
* NVIDIA CUDA Toolkit
* `nvcc` CUDA compiler

Individual projects can be compiled using `nvcc`. For example:

```bash
nvcc template.cu -o template
./template
```
