//
//  IBM Corporation (C) 2019
//  Nelson Mimura -- nmimura@ibm.com
//

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

//
//  defaults
//

#define DEFAULT_PPN        6
#define DEFAULT_LEN        1024
#define DEFAULT_BLOCK_SIZE 512
#define DEFAULT_REPS       1024
#define DEFAULT_STREAMS    4

#define EPSILON 1.0E-6

//
//  finalize program
//

void finalize(
        int rc) // program return code
{
    if (rc) {
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    else {
        MPI_Finalize();
        exit(rc);
    }
}

//
//  print error message and exit
//

void fail(
        const char* fmt,    // message string (printf-like format)
        ...)                // additional parameters
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    finalize(EXIT_FAILURE);
}

//
//  check CUDA call
//

void assertCuda(
        cudaError_t rc,
        const char* file,
        int line)
{
    if (rc)
        fail("[error] %s@%d: CUDA Runtime error %d: %s\n",
                file, line,
                rc, cudaGetErrorString(rc));
}

#define checkCuda(x) assertCuda((x), __FILE__, __LINE__)

//
//  timing
//

double clock_diff(
        struct timespec* t1,
        struct timespec* t0)
{
    struct timespec dt;

    if (t1->tv_nsec >= t0->tv_nsec) {
        dt.tv_sec  = t1->tv_sec  - t0->tv_sec;
        dt.tv_nsec = t1->tv_nsec - t0->tv_nsec;
    }
    else {
        dt.tv_sec  = t1->tv_sec  - t0->tv_sec  - 1;
        dt.tv_nsec = t1->tv_nsec - t0->tv_nsec + 1E9;
    }

    return (double) dt.tv_sec + (double) dt.tv_nsec / 1.0E9;
}

//
//  program kernel
//

__global__ void kernel(
        double* dx,
        size_t len,
        size_t reps)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < len) {
        dx[i] = (double) i;
        for (size_t r = 0; r < reps; r++)
            dx[i]++;
    }
}

//
//  print usage information and exit
//

void usage(
        const char* exec)   // program name/path
{
    fprintf(stderr,
            "usage: %s\n"
            "optional parameters:\n"
            "   -p NUM      processes per resource set\n"
            "   -l NUM      array length\n"
            "   -r NUM      kernel repetitions\n"
            "   -s NUM      number of streams\n"
            "   -c          check results\n"
            , exec);

    finalize(EXIT_SUCCESS);
}

//
// program entry point
//

int main(
        int argc,           // number of arguments
        char** argv)        // argument list
{
    int rank;               // my rank id
    int ranks;              // number of ranks
    int ppn;                // processes per node
    int device;             // device number to use

    size_t len;             // array length (elements)
    size_t size;            // array size (bytes)
    double* hx;             // data array (host)
    double* dx;             // data array (device)
    size_t len_s;           // length per stream
    size_t size_s;          // size per stream
    size_t off;             // offset for operations

    size_t reps;            // kernel internal repetitions
    size_t streams;         // number of streams
    cudaStream_t* stream;   // array of streams

    size_t tpb;             // threads per block
    size_t bpg;             // blocks per grid

    int check = 0;          // check results
    
    struct timespec t0, t1; // basic timing

    ppn       = DEFAULT_PPN;
    len       = DEFAULT_LEN;
    reps      = DEFAULT_REPS;
    streams   = DEFAULT_STREAMS;
    tpb       = DEFAULT_BLOCK_SIZE;

    // start
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    // parse arguments
    for (int i = 1; i < argc; i++) {
        switch (argv[i][1]) {
        case 'h': usage(argv[0]);
        case 'p': ppn     = atoi(argv[++i]); break;
        case 'l': len     = atol(argv[++i]); break;
        case 'r': reps    = atol(argv[++i]); break;
        case 's': streams = atoi(argv[++i]); break;
        case 'c': check   = 1;               break;
        default:
            fail("error: invalid parameter: '%s'\n",
                    argv[i]);
        }
    }

    // set device
    device = rank % ppn;
    checkCuda(cudaSetDevice(device));

    // check data
    if (len % streams)
        fail("error: length '%lu' not divisible by streams '%d'\n",
                len, streams);

    // allocate data
    size = len * sizeof(double);
    checkCuda(cudaMallocHost(&hx, size));
    checkCuda(cudaMalloc(&dx, size));
    len_s = len / streams;
    size_s = size / streams;

    // create streams
    stream = (cudaStream_t*) malloc(streams * sizeof(cudaStream_t));
    for (size_t s = 0; s < streams; s++)
        checkCuda(cudaStreamCreate(&stream[s]));

    // launch setup
    bpg = (len + (tpb - 1)) / tpb;

    // run test
    MPI_Barrier(MPI_COMM_WORLD);
    checkCuda(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (size_t s = 0; s < streams; s++) {
        off = s * len_s;
        checkCuda(cudaMemcpyAsync(&dx[off], &hx[off], size_s,
                    cudaMemcpyHostToDevice, stream[s]));
        kernel<<<bpg, tpb, 0, stream[s]>>>(dx, len, reps);
        checkCuda(cudaStreamSynchronize(stream[s]));
        checkCuda(cudaMemcpyAsync(&hx[off], &dx[off], size_s,
                    cudaMemcpyDeviceToHost, stream[s]));
    }

    // wait for all streams
    for (size_t s = 0; s < streams; s++)
        checkCuda(cudaStreamSynchronize(stream[s]));
    checkCuda(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    // check results
    if (check)
        for (size_t i = 0; i < len; i++)
            if (fabs(hx[i] - (i + reps) >= EPSILON))
                fail("error: check %lu failed: %lf\n", i, hx[i]);

    // cleanup and finish
    for (size_t s = 0; s < streams; s++)
        checkCuda(cudaStreamDestroy(stream[s]));
    free(stream);

    if (!rank)
        fprintf(stdout,
                "success! runtime: %lf\n", clock_diff(&t1, &t0));
    finalize(EXIT_SUCCESS);
}

