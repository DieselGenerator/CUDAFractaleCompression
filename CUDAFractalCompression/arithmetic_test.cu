#include <iostream>
#include <chrono>
#include <inttypes.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "arithmetic_test.cuh"

#include "cuda_call_checker.cuh"

__global__ void abs1(int* d_forabs){

	int x_tid = threadIdx.x;
	int y_tid = threadIdx.y;
	int my_abs = x_tid - y_tid;

	volatile int i = my_abs < 0 ? -my_abs : my_abs;
}


__global__ void abs2(int* d_forabs) {

	int x_tid = threadIdx.x;
	int y_tid = threadIdx.y;
	int my_abs = x_tid - y_tid;

	volatile int i = (my_abs ^ (my_abs >> 31)) - (my_abs >> 31);

}

__global__ void max() {
	uint8_t ui8 = 0;
	double d = 1000;
	ui8 = (uint8_t)d;
	printf("uint8_t value : %d \n", ui8);
}

void arithmetic_speedtest(){
	
	CHECK(cudaDeviceReset());

	std::cout << "size ui" <<  sizeof(unsigned int) << std::endl;
	std::chrono::system_clock::time_point start, end;
	int forabs[64];
	int *d_forabs;
	//33554432,64,64
	dim3 grid(10000, 64, 64);
	dim3 block(8, 8);
	
	//dim3 grid(1);
	//dim3 block(32, 32, 2);
	CHECK(cudaMalloc((void**)&d_forabs, sizeof(int) * 64));

	start = std::chrono::system_clock::now();

	abs1<<<grid, block>>>(d_forabs);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	CHECK(cudaDeviceSynchronize());

	end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "abs1 elapsed " << elapsed << " milli sec \n";

	CHECK(cudaMemcpy(forabs, d_forabs, sizeof(int) * 64, cudaMemcpyDeviceToHost));
	CHECK(cudaDeviceSynchronize());

	//abs2

	start = std::chrono::system_clock::now();

	abs2<<<grid, block>>>(d_forabs);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	CHECK(cudaDeviceSynchronize());

	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "abs2 elapsed " << elapsed << " milli sec \n";

	CHECK(cudaMemcpy(forabs, d_forabs, sizeof(int) * 64, cudaMemcpyDeviceToHost));
	CHECK(cudaDeviceSynchronize());

	//max
	max<<<1, 1>>>();
	CHECK(cudaDeviceSynchronize());
	



	//for(int i = 0; i < 64; i++){
	//	std::cout << forabs[i] << std::endl;
	//}
}
