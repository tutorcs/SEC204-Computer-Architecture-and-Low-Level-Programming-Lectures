https://tutorcs.com
WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com

#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <Windows.h>

#define M 8000

__declspec(align(64)) float  X[M], Y[M], A[M][M], X2[M], Y2[M];

void initialization();
void MVM_default();
void MVM_SSE();
void MVM_AVX();

void MMM_default();

void FIR_default();

using namespace std;

int main() {

	//the following command pins the current process to the 1st core
	//otherwise, the OS tongles this process between different cores
	BOOL success = SetProcessAffinityMask(GetCurrentProcess(), 1);
	if (success==0) 	{
		cout << "SetProcessAffinityMask failed" << endl; 
		system("pause");
		return -1;
	}

	//initialize the arrays
	initialization();

	//define the timers measuring execution time
	//clock_t start_1, end_1;

	//start_1 = clock();
	auto start = std::chrono::high_resolution_clock::now();

	//run this 10 times because this routine runs very fast 
	//The execution time needs to be at least some seconds in order to have a good measurement (why?) 
	//			because other processes run at the same time too, preempting our thread
	for (int t = 0; t < 20; t++) {
		MVM_default();
		//MVM_SSE();
		//MVM_AVX();
	}



	auto finish = std::chrono::high_resolution_clock::now();
	//end_1 = clock();

	//printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";

	system("pause");
	return 0;
}


void initialization() {

	for (int i = 0; i != M; i++)
		for (int j = 0; j != M; j++)
			A[i][j] = (float)(i - j);

	for (int j = 0; j != M; j++) {
		Y[j] = 0.0;
		Y2[j] = 0.0;
		X[j] = (float)j;
		X2[j] = (float)j;
	}
}

void MVM_default() {

	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			Y[i] += A[i][j] * X[j];

}

void MVM_SSE() {

	__m128 num0, num1, num2, num3, num4, num5, num6;

	for (int i = 0; i < M; i++) {

		num3 = _mm_setzero_ps();
		for (int j = 0; j < M; j += 4) {

			num0 = _mm_load_ps(&A[i][j]);
			num1 = _mm_load_ps(X + j);
			num3 = _mm_fmadd_ps(num0, num1, num3);
		}

		num4 = _mm_hadd_ps(num3, num3);
		num4 = _mm_hadd_ps(num4, num4);

		_mm_store_ss((float *)Y + i, num4);
	}
}

void MVM_AVX() {

	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, num0, num1, num2, num3, num4, num5;
	__m128 xmm1, xmm2;

	for (int i = 0; i < M; i++) {
		num1 = _mm256_setzero_ps();

		for (int j = 0; j < M; j += 8) {

			num5 = _mm256_load_ps(X + j);
			num0 = _mm256_load_ps(&A[i][j]);
			num1 = _mm256_fmadd_ps(num0, num5, num1);
		}

		//xmm1 = _mm_load_ss(Y + i);
		ymm2 = _mm256_permute2f128_ps(num1, num1, 1);
		num1 = _mm256_add_ps(num1, ymm2);
		num1 = _mm256_hadd_ps(num1, num1);
		num1 = _mm256_hadd_ps(num1, num1);
		xmm2 = _mm256_extractf128_ps(num1, 0);
		//xmm2 = _mm_add_ps(xmm2, xmm1);
		_mm_store_ss((float *)Y + i, xmm2);
	}

}


//arrays need to be defined by you...
void MMM_default() {
	/*
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			for (int k = 0; k < M; k++)
				C[i][j] += A[i][k] * B[k][j];
				*/
}

//arrays need to be defined by you...
//float out[N],in[N+M],kernel[M];
void FIR_default() {
	/*
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			out[i] += in[i + j] * kernel[j];
			*/

}

