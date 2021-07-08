//
//  IBM Corporation (C) 2019
//  Nelson Mimura -- nmimura@ibm.com
//

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define gettid() syscall(SYS_gettid)

#include <unordered_map>

#include <mpi.h>
#include <cuda_runtime.h>

//
//  CONSTANTS
//  ----------------------------------------------------------------------------
//

#define LWP_HOST 128    // hostname length
#define LWP_NAME 128    // file name/path length

//
//  STRUCTURES
//  ----------------------------------------------------------------------------
//

//
//  event
//

typedef struct {
    size_t function;        // function
    struct timespec t0;     // API start timestamp
    struct timespec t1;     // API end timestamp
    cudaEvent_t e0;         // event start
    cudaEvent_t e1;         // event end
    cudaStream_t stream;    // stream
} ev_t;

//
//  thread-local global state
//

typedef struct {
    char host[LWP_HOST];    // hostname
    pid_t pid;              // process id
    pid_t tid;              // thread id
    int rank;               // my rank id
    int ranks;              // number of ranks
    int digits;             // rank formatting
    int debug;              // enable debug messages
    size_t evm;             // max events
    size_t evc;             // event counter
    ev_t* evs;              // events
    struct timespec tb;     // base timestamp
    cudaEvent_t eb;         // base event
} var_t;

__thread var_t var;
__thread int lwp_initialized = 0;

void lwp_init();
void lwp_fini();

//
//  UTILITIES
//  ----------------------------------------------------------------------------
//

//
//  log message
//

#define log(fmt, ...) \
    fprintf(stderr, "[lwp] " fmt, ##__VA_ARGS__)

#define fail(fmt, ...) \
    { log(fmt, ##__VA_ARGS__); exit(EXIT_FAILURE); }

//
//  retrieve environment value
//

const char* env(
        const char* envar,
        const char* defval)
{
    const char* enval;  // env value

    enval = getenv(envar);
    if (!enval)
        enval = defval;

    return enval;
}

//
//  calculate time difference
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
        dt.tv_sec  = t1->tv_sec  - t0->tv_sec  + 1;
        dt.tv_nsec = t1->tv_nsec - t0->tv_nsec - 1E9;
    }

    return (double) dt.tv_sec + (double) dt.tv_nsec / 1.0E9;
}

