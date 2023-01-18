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



//MVM initialization 
#define M 8000
__declspec(align(64)) float  X[M], Y[M], A1[M][M];

//MMM initialization 
#define N 1024
__declspec(align(64)) float  C[N][N], A[N][N], B[N][N], Btranspose[N][N];

//FIR initialization 
#define num 4  //where num=1,2,3,4,5,...
#define M1 4000*num  
#define N1 64032*num
__declspec(align(64)) float out[N1],in[N1+M1],kernel[M1];


void initialization();
void MVM_default();
void MVM_SSE();
void MVM_AVX();

void MMM_default();
void MMM_SSE();
void MMM_AVX();

void FIR_default();
void FIR_SSE();
void FIR_AVX();

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

	//run this several times because this routine runs very fast 
	//The execution time needs to be at least some seconds in order to have a good measurement (why?) 
	//			because other processes run at the same time too, preempting our thread
	//for MMM and FIR you can use t=1
	for (int t = 0; t < 20; t++) {
		MVM_default();
		//MVM_SSE();
		//MVM_AVX();

		//MMM_default();
		//MMM_SSE();
		//MMM_AVX();

		//FIR_default();
		//FIR_SSE();
		//FIR_AVX();
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

	//MVM
	for (unsigned int i = 0; i != M; i++)
		for (unsigned int j = 0; j != M; j++)
			A1[i][j] = (float)(i - j);

	for (unsigned int j = 0; j != M; j++) {
		Y[j] = 0.0;
		X[j] = (float)j;
	}

//MMM
for (unsigned int i=0;i<N;i++){ //printf("\n");
for (unsigned int j=0;j<N;j++){
  C[i][j]=0.0;
  A[i][j]=(float) (j%23); //printf(" %3.1f",A[i][j]);
  B[i][j]=(float) (j%41); //printf(" %3.1f",B[i][j]);
}
}


//FIR
for(unsigned int i = 0; i != N1+M1 ; i++ )
in[i]=(float) (i%17);

for(unsigned int i = 0; i != M1 ; i++ )
kernel[i]=(float)  (i%21);

}

void MVM_default() {

	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			Y[i] += A1[i][j] * X[j];

}

void MVM_SSE() {

	__m128 num0, num1, num2, num3, num4, num5, num6;

	for (int i = 0; i < M; i++) {

		num3 = _mm_setzero_ps();
		for (int j = 0; j < M; j += 4) {

			num0 = _mm_load_ps(&A1[i][j]);
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
			num0 = _mm256_load_ps(&A1[i][j]);
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



void MMM_default() {
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
				C[i][j] += A[i][k] * B[k][j];
				
}



void MMM_SSE() {
	int i, j, k;
	__m128 num0, num1, num2, num3, num4, num5, num6, num7;

	//B[][] array needs to be written in maim memory column-wise and not row-wise
	//SIMD hardware can load only consecutive main memory locations and the arrays are written in main memory row-wise
	//thus, this loop kernels copies the data from B[][] and stores them to Btranspose[][]
	//this loop kernel can be implemented using SSE too, thus boosting performance
	for (j = 0; j != N; j++)
		for (k = 0; k != N; k++) {
			Btranspose[k][j] = B[j][k];
		}

     //this is not the most efficient implementation but it is faster 
	for (i = 0; i != N; i++)
		for (j = 0; j != N; j++) {
			num3 = _mm_setzero_ps();
			for (k = 0; k != N; k += 4) {
				num0 = _mm_load_ps(&A[i][k]);
				num1 = _mm_load_ps(&Btranspose[j][k]);
				num3 = _mm_fmadd_ps(num0, num1, num3);
			}
			num4 = _mm_hadd_ps(num3, num3);
			num4 = _mm_hadd_ps(num4, num4);
			_mm_store_ss((float *)&C[i][j], num4);
		}
}



void MMM_AVX() {
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
	int i, j, k;

	//B[][] array needs to be written in maim memory column-wise and not row-wise
//SIMD hardware can load only consecutive main memory locations and the arrays are written in main memory row-wise
//thus, this loop kernels copies the data from B[][] and stores them to Btranspose[][]
//this loop kernel can be implemented using SSE too, thus boosting performance
	for (j = 0; j != N; j++)
		for (k = 0; k != N; k++) {
			Btranspose[k][j] = B[j][k];
		}

	//this is not the most efficient implementation but it is faster
	for (i = 0; i != N; i++)
		for (j = 0; j != N; j++) {
			ymm0 = _mm256_setzero_ps();
			for (k = 0; k != N; k += 8) {
				ymm1 = _mm256_load_ps(&A[i][k]);
				ymm2 = _mm256_load_ps(&Btranspose[j][k]);
				ymm0 = _mm256_fmadd_ps(ymm1, ymm2, ymm0);
			}

			ymm2 = _mm256_permute2f128_ps(ymm0, ymm0, 1);
			ymm0 = _mm256_add_ps(ymm0, ymm2);
			ymm0 = _mm256_hadd_ps(ymm0, ymm0);
			ymm0 = _mm256_hadd_ps(ymm0, ymm0);
			_mm_store_ss((float *)&C[i][j], _mm256_extractf128_ps(ymm0, 0));
		}

}




void FIR_default() {
	
	for (int i = 0; i < N1; i++)
		for (int j = 0; j < M1; j++)
			out[i] += in[i + j] * kernel[j];
			

}

void FIR_SSE() {
	int i, j;
	__m128 num0, num1, num2, num3, num4, num5, num6;


	for (i = 0; i != N1; i++) {
		num1 = _mm_setzero_ps();
		for (j = 0; j != M1; j += 4) {
			num5 = _mm_load_ps(kernel + j);

			num0 = _mm_loadu_ps(in + i + j);
			num1 = _mm_fmadd_ps(num0, num5, num1);
		}
		num4 = _mm_hadd_ps(num1, num1);
		num4 = _mm_hadd_ps(num4, num4);
		_mm_store_ss((float *)out + i, num4);
	}
}


void FIR_AVX() {
	int i, j;
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
	__m256 num0, num1, num2, num3, num4, num5, num6, num7;
	__m128 xmm1, xmm2;

	for (i = 0; i != N1; i++) {
		num1 = _mm256_setzero_ps();
		for (j = 0; j != M1; j += 8) {
			num5 = _mm256_load_ps(kernel + j);

			num0 = _mm256_loadu_ps(in + i + j);
			num1 = _mm256_fmadd_ps(num0, num5,num1);
		}

		ymm2 = _mm256_permute2f128_ps(num1, num1, 1);
		num1 = _mm256_add_ps(num1, ymm2);
		num1 = _mm256_hadd_ps(num1, num1);
		num1 = _mm256_hadd_ps(num1, num1);
		xmm2 = _mm256_extractf128_ps(num1, 0);
		_mm_store_ss((float *)out + i, xmm2);

	}
}

