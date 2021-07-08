//
//  IBM Corporation (C) 2019
//  Nelson Mimura -- nmimura@ibm.com
//

//
//  BASE
//  ----------------------------------------------------------------------------
//

#include "lwp_base.cpp"

//
//  ENVIRONMENT
//  ----------------------------------------------------------------------------
//

//
//  env variables
//

#define ENVAR_RANK  "OMPI_COMM_WORLD_RANK"
#define ENVAR_RANKS "OMPI_COMM_WORLD_SIZE"
#define ENVAR_DEBUG "LWP_DEBUG"
#define ENVAR_EVM   "LWP_MAX"

//
//  env defaults
//

#define DEFVAL_RANK  "0"
#define DEFVAL_RANKS "1"
#define DEFVAL_DEBUG "0"
#define DEFVAL_EVM   "1024"

//
//  WRAPPERS
//  ----------------------------------------------------------------------------
//

#include "lwp_wrappers.cpp"

//
//  INITIALIZE
//  ----------------------------------------------------------------------------
//

__attribute__((constructor)) void lwp_init()
{
    gethostname(var.host, LWP_HOST);
    var.pid = getpid();
    var.tid = gettid();

    var.rank  = atoi(env(ENVAR_RANK,  DEFVAL_RANK));
    var.ranks = atoi(env(ENVAR_RANKS, DEFVAL_RANKS));

    var.digits = 0;
    for (int i = var.ranks; i > 0; i /= 10)
        var.digits++;

    var.debug = atoi(env(ENVAR_DEBUG, DEFVAL_DEBUG));

    if (var.debug)
        log("started: %s %d %d %0*d\n",
                var.host, var.pid, var.tid, var.digits, var.rank);

    var.evm = atol(env(ENVAR_EVM,   DEFVAL_EVM));
    var.evc = 0;
    var.evs = (ev_t*) malloc(var.evm * sizeof(ev_t));

    if (!var.rank && var.debug) {
        log("max events: %lu\n", var.evm);
        log("size per event: %lu\n", sizeof(ev_t));
        log("memory footprint: %lu MB\n",
                ( sizeof(var_t) + var.evm * sizeof(ev_t)) / 1024 / 1024);
    }

    for (size_t i = 0; i < var.evm; i++) {
        cudaEventCreate(&var.evs[i].e0);
        cudaEventCreate(&var.evs[i].e1);
    }

    lwp_initialized = 1;
}

//
//  FINALIZE
//  ----------------------------------------------------------------------------
//

__attribute__((destructor)) void lwp_fini()
{
    char fn[LWP_NAME] = {0};    // file name
    FILE* fp;                   // file pointer

    double dt;  // clock delta
    float ms;   // event delta

    std::unordered_map<size_t, char*> fmap; // address -> name

    cudaDeviceSynchronize();

    if (var.debug)
        log("rank %0*d recorded events: %lu\n",
                var.digits, var.rank, var.evc);

    sprintf(fn, "lwp.%0*d.%s.%d.%d", 
            var.digits, var.rank,
            var.host, var.pid, var.tid);
    fp = fopen(fn, "w");

    if (!fp)
        fail("rank %0*d could not create output file: error %d: %s\n",
                var.digits, var.rank, errno, strerror(errno));

    fwrite(&var, sizeof(var_t), 1, fp);

    for (size_t i = 0; i < var.evc; i++) {
        ev_t* ev = &var.evs[i];
        fwrite(&ev->function, sizeof(size_t), 1, fp);

        dt = clock_diff(&ev->t0, &var.tb);
        fwrite(&dt, sizeof(double), 1, fp);
        dt = clock_diff(&ev->t1, &var.tb);
        fwrite(&dt, sizeof(double), 1, fp);

        cudaEventElapsedTime(&ms, var.eb, ev->e0);
        dt = ms / 1.0E3;
        fwrite(&dt, sizeof(double), 1, fp);
        cudaEventElapsedTime(&ms, var.eb, ev->e1);
        dt = ms / 1.0E3;
        fwrite(&dt, sizeof(double), 1, fp);

        if (!fmap.count(ev->function)) {
            Dl_info info;
            if (!dladdr((void*) ev->function, &info))
                fail("error: "
                     "address could not be matched to a shared object: "
                     "0x%016lx\n",
                     (size_t) ev->function);

            const char* name = info.dli_sname;
            if (!name) {
                log("warning: "
                     "address could not be matched to a symbol: "
                     "0x%016lx\n",
                     (size_t) ev->function);
                name = "cudaLaunch(Kernel)";
            }

            size_t len = strlen(name);
            char* copy = (char*) malloc(len + 1);
            memset(copy, 0, len + 1);
            strcpy(copy, name);
            fmap[ev->function] = copy;
        }
    }

    size_t fmap_n = fmap.size();
    fwrite(&fmap_n, sizeof(size_t), 1, fp);
    
    std::unordered_map<size_t, char*>::iterator it;
    for (it = fmap.begin(); it != fmap.end(); it++) {
        size_t len = strlen(it->second);
        fwrite(&it->first, sizeof(size_t), 1, fp);
        fwrite(&len, sizeof(size_t), 1, fp);
        fwrite(it->second, sizeof(char), len + 1, fp);
    }

    fclose(fp);
}

