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

#define N 16000
#define Tile 8


__declspec(align(64)) float  X[N], A[N][N];

//__declspec(align(64)) float  C[N][N], Btranspose[N][N];


void row_column_wise();
void cache_benchmark(const int &i);
void initialization();

using namespace std;

int main() {

	//the following command pins the current process to the 1st core
	//otherwise, the OS tongles this process between different cores
	BOOL success = SetProcessAffinityMask(GetCurrentProcess(), 1);
	if (success == 0) {
		cout << "SetProcessAffinityMask failed" << endl;
		system("pause");
		return -1;
	}

	initialization();

	clock_t start_1, end_1;

	//start_1 = clock();
	auto start = std::chrono::high_resolution_clock::now();


	//run this 10 times because this routine runs very fast 
	//The execution time needs to be at least some seconds in order to have a good measurement (why?) 
	//			because other processes run at the same time too, preempting our thread
	for (int t = 0; t < 100000; t++) {
		//row_column_wise();
		cache_benchmark(t);
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

	for (int i = 0; i != N; i++)
		for (int j = 0; j != N; j++)
			A[i][j] = (float)(i - j);

	for (int j = 0; j != N; j++) {
		X[j] = (float)j;
	}
}

//comment/uncomment the appropriate loop kernel below
void row_column_wise() {

	//row-wise initialization
	for (int i = 0; i != N; i++)
		for (int j = 0; j != N; j++)
			A[i][j] = i + j;


	//column-wise initialization
	/*
	for (int j = 0; j != N; j++)
		for (int i = 0; i != N; i++)
			A[i][j] = i + j;
			*/


}


//5.3 , 10.6 , 21 , 42
void cache_benchmark(const int &i) {

		for (int j = 0; j < N; j++) {
			X[j] = (float) i;
		}

}
