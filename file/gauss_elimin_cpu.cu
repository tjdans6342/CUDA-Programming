
#include <bits/stdc++.h>
#include <time.h>
#include <sys/time.h>
using namespace std;
#define N 20

void make_primary_matrix(double arr[], int n) {
	for (int i=0; i<n; i++) {
		for (int j=i; j<n; j++) {
			arr[i*(n+1)+j] = arr[j*(n+1)+i] = i+1;
		}
	}
	for (int i=0; i<n; i++) {
		int num = 0;
		for (int j=0; j<n; j++) {
			num += arr[i*(n+1)+j];
		}
		arr[i*(n+1)+n] = num;
	}
}

void swap_arr(double arr[], int col1, int col2, int n) {
	for (int i=0; i<n+1; i++) {
		swap(arr[(n+1)*col1 + i], arr[(n+1)*col2 + i]);
	}
}

void gaussElimin(double arr[], double *x, int n) {
	for (int j=0; j<n-1; j++) {
		if (abs(arr[j*(n+1)+j] - 0.0) < 1e-10) {
			for (int k=j+1; k<n; k++) {
				if (abs(arr[k*(n+1)+j] - 0.0) > 1e-10) {
					swap_arr(arr, k, j, n);
					break;
				}
			}
		}

		for (int i=j+1; i<n; i++) {
			double lam = arr[i*(n+1)+j] / arr[j*(n+1)+j];
			for (int k=0; k<n; k++) {
				arr[i*(n+1)+k] -= lam*arr[j*(n+1)+k];
			}
			arr[i*(n+1)+n] -= lam*arr[j*(n+1)+n];
		}
	}

	// 역대입법
	x[n-1] = arr[(n-1)*(n+1)+n] / arr[(n-1)*(n+1)+n-1];

	for (int i=n-2; i>=0; i--) {
		double val = 0;
		for (int k=i+1; k<n; k++) {
			val += x[k] * arr[i*(n+1)+k];
		}
		x[i] = (arr[i*(n+1)+n] - val) / arr[i*(n+1)+i];
	}

}

int main()
{	
    struct timeval start, end;


	double *arr, *x;
	arr = (double*) malloc((N)*(N+1)*sizeof(double));
    x = (double*) malloc(N*sizeof(double));

    printf("size of matrix:  %d\n", N);

    make_primary_matrix(arr, N);

    gettimeofday(&start, NULL);
    gaussElimin(arr, x, N);
    gettimeofday(&end, NULL);

    double start_micro = (double)start.tv_sec*1000000 + (double)start.tv_usec;
    double end_micro = (double)end.tv_sec*1000000 + (double)end.tv_usec;

    printf("\n%lf\n", (end_micro-start_micro)/1000000);

	return 0;
}
