
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#define N 20

#define THREADS_PER_BLOCK 16

// Host function -------------------------------------------------------------------
double *malloc_matrix(const int a, const int b) {
    return (double*)malloc(sizeof(double *)*a*b);
}

double *make_primary_matrix(int *a_size) { // 해가 유일한 행렬 만들기
    
    int size = N;
    double *matrix_arr = malloc_matrix(size, size + 1);

    for (int i=0; i<N; i++) {
        for (int j=i; j<N; j++) {
            matrix_arr[i*(N+1)+j] = matrix_arr[j*(N+1)+i] = i+1;
        }
    }
    for (int i=0; i<N; i++) {
        int num = 0;
        for (int j=0; j<N; j++) {
            num += (j+1)*matrix_arr[i*(N+1)+j];
        }
        matrix_arr[i*(N+1)+N] = num;
    }

    *a_size = size;

    return matrix_arr;
}
// --------------------------------------------------------------------------------------


// Device function -------------------------------------------------------------------
__global__ void replace_zero_gpu(double *d_arr, int rows, int columns, int column) {
    if(fabs(d_arr[column*columns + column]) <= 1e-4) {
        int row = column;
        for(; row < rows; row++) {
            if(fabs(d_arr[row*columns + column]) > 1e-4)
                break;
        }
        int tidx= blockDim.x*blockIdx.x + threadIdx.x;
        if(tidx+ column >= columns)
            return;

        __syncthreads();
        int zero = column*columns + column + tidx; // x축 이동
        int chosen = row*columns + column + tidx; // x축 이동
        d_arr[zero] += d_arr[chosen];
    }
}

__global__ void column_elimination_gpu(double *d_arr, int rows, int columns, int col) {
    int tidx= blockDim.x*blockIdx.x + threadIdx.x;
    if(tidx >= (rows - 1 - col)*(columns - col)) // columns 만큼의 thread를 사용하지 x
        return;

    int sub_y = tidx/(columns-col);
    int sub_x = tidx%(columns-col);
    int gl_y = col+1 + sub_y;
    int gl_x = col + sub_x;

    int gl_idx = gl_x + gl_y*columns;
    int up_idx = gl_x + col*columns;

    int up_el = col + col*columns;
    int gl_el = col + gl_y*columns;
    double lam = d_arr[gl_el]/d_arr[up_el];

    d_arr[gl_idx] -= lam*d_arr[up_idx];
}

__global__ void multiple_column(double *d_arr, int rows, int columns, int row) {
    int tidx= threadIdx.x;

    int cols = columns - 2 - row; // 바꿔야 하는 개수

    int start_index_cols = row*columns + row;
    int end_index_rows = rows*columns - 1;

    d_arr[start_index_cols + tidx+1] *= d_arr[end_index_rows - columns*(cols-1-tidx)];
}

__global__ void reverse_row_elimination(double *d_arr, int rows, int columns, int row) {
    int tidx= threadIdx.x;
    int cols = columns - 2 - row;

    int start_index = row*columns + row; // (row, row) 좌표

    for (int i=cols; i>=2; i/=2) {
        bool is_odd;
        if (i%2 == 1) is_odd = true;
        else is_odd = false;

        int step = i/2;
        if (tidx>= step) return;

        d_arr[start_index + tidx+1] += (d_arr[start_index + tidx+1 + step]); 
        d_arr[start_index + tidx+1 + step] = 0;

        if (is_odd && tidx+1 == step) {
            d_arr[start_index + tidx+1] += d_arr[start_index + tidx+1 + step+1];
            d_arr[start_index + tidx+1 + step+1] = 0;
        }

        __syncthreads();
    }

    int x_el = (row + 1)*columns - 1;
    int diag_el = row*columns + row;

    if(diag_el + 1 != x_el) {
        d_arr[x_el] -= d_arr[diag_el + 1];
        d_arr[diag_el + 1] = 0.0;
    }

    d_arr[x_el] /= d_arr[diag_el];
    d_arr[diag_el] = 1.0;
}

// -----------------------------------------------------------------------------------


// Host function -------------------------------------------------------------------
void start_gaussian_elimination_gpu(double *arr, int rows, int cols) {
    double *dev_arr;

    cudaMalloc(&dev_arr, sizeof(double)*rows*cols);
    cudaMemcpy(dev_arr, (void*)arr, sizeof(double)*rows*cols, cudaMemcpyHostToDevice);

    int block_size;

    for(int y=0; y<cols-1; y++) {
        block_size = (cols-y-1)/THREADS_PER_BLOCK + 1;
        replace_zero_gpu <<<block_size, THREADS_PER_BLOCK>>> (dev_arr, rows, cols, y);

        block_size = ((rows-1 - y )*(cols - y) - 1)/THREADS_PER_BLOCK + 1;
        column_elimination_gpu <<<block_size, THREADS_PER_BLOCK>>> (dev_arr, rows, cols, y);
    }

    for(int x=rows-1; x>=0; x--) {

        multiple_column<<<1, cols-2-x>>>(dev_arr, rows, cols, x);
        reverse_row_elimination<<<1, cols>>>(dev_arr, rows, cols, x);

    }

    cudaMemcpy(arr, (void*)dev_arr, sizeof(double)*rows*cols, cudaMemcpyDeviceToHost);
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            printf("%.2f ", arr[i*cols+j]);
        }
        printf("\n");
    }
    printf("\n");
    
    cudaFree(dev_arr);
}


int main(int argc, char ** argv) {
    struct timeval start, end;

    int size;
    double *arr = make_primary_matrix(&size);

    printf("size of matrix:  %d\n", N);

    gettimeofday(&start, NULL);
    start_gaussian_elimination_gpu(arr, size, size + 1);
    gettimeofday(&end, NULL);


    double start_micro = (double)start.tv_sec*1000000 + (double)start.tv_usec;
    double end_micro = (double)end.tv_sec*1000000 + (double)end.tv_usec;

    printf("\n%lf\n", (end_micro-start_micro)/1000000);

    return 0;
}
// ---------------------------------------------------------------------------------------
