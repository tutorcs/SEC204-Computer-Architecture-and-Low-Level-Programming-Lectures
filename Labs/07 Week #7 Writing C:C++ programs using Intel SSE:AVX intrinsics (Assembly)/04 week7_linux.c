https://tutorcs.com
WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
//compile with:
//gcc week7_linux.c -o p -march=native -mavx2 -lm -D_GNU_SOURCE  -g  -pthread -O3


#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>

#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>

#define M 8000

float  X[M], Y[M], A[M][M], X2[M], Y2[M] __attribute__((aligned(64)));

void initialization();
void MVM_default();
void MVM_SSE();
void MVM_AVX();

void MMM_default();

void FIR_default();


int main() {

int t;
time_t start1, end1;
struct timeval start2, end2;

	//the following command pins the current process to the 1st core
	//otherwise, the OS tongles this process between different cores
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(1,&mask);
	if(sched_setaffinity(0,sizeof(mask),&mask) == -1)
       		printf("WARNING: Could not set CPU Affinity, continuing...\n");

	//initialize the arrays
	initialization();

	//define the timers measuring execution time
	start1 = clock();
	gettimeofday(&start2, NULL);


	//run this 10 times because this routine runs very fast 
	//The execution time needs to be at least some seconds in order to have a good measurement (why?) 
	//			because other processes run at the same time too, preempting our thread
	for (t = 0; t < 20; t++) {
		MVM_default();
		//MVM_SSE();
		//MVM_AVX();
	}



	end1 = clock();
	gettimeofday(&end2, NULL);
	printf(" clock() method: %ldms\n", (end1 - start1) / (CLOCKS_PER_SEC/1000));
	printf(" gettimeofday() method: %ldms\n", (end2.tv_sec - start2.tv_sec) *1000 + (end2.tv_usec - 	start2.tv_usec)/1000);

	return 0;
}


void initialization() {

int i,j;

	for (i = 0; i != M; i++)
		for (j = 0; j != M; j++)
			A[i][j] = (float)(i - j);

	for (j = 0; j != M; j++) {
		Y[j] = 0.0;
		Y2[j] = 0.0;
		X[j] = (float)j;
		X2[j] = (float)j;
	}
}

void MVM_default() {

int i,j;

	for (i = 0; i < M; i++)
		for (j = 0; j < M; j++)
			Y[i] += A[i][j] * X[j];

}

void MVM_SSE() {

int i,j;
	__m128 num0, num1, num2, num3, num4, num5, num6;

	for (i = 0; i < M; i++) {

		num3 = _mm_setzero_ps();
		for (j = 0; j < M; j += 4) {

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
	int i,j;
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, num0, num1, num2, num3, num4, num5;
	__m128 xmm1, xmm2;

	for (i = 0; i < M; i++) {
		num1 = _mm256_setzero_ps();

		for (j = 0; j < M; j += 8) {

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


