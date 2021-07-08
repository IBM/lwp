//
//  IBM Corporation (C) 2019
//  Nelson Mimura -- nmimura@ibm.com
//

#include <stdio.h>

#include "lwp_base.cpp"

#define W 13 // time width
#define P  9 // time precision

void process(
        const char* fpath)
{
    FILE* fp;       // input file pointer
    var_t var;      // rank state
    size_t size;    // event actual size
    
    size_t fmap_n;                          // map entries
    std::unordered_map<size_t, char*> fmap; // function map
    size_t function;                        // function address
    int max = 0;                            // longest function (name)

    fp = fopen(fpath, "r");
    fread(&var, sizeof(var_t), 1, fp);
   
    size = sizeof(size_t) + 4 * sizeof(double);
    fseek(fp, var.evc * size, SEEK_CUR);
    fread(&fmap_n, sizeof(size_t), 1, fp);

#if _DEBUG
    log("%0*d #events: %lu\n", var.digits, var.rank, var.evc);
    log("%0*d #base: %ld.%09ld\n",
            var.digits, var.rank,
            var.tb.tv_sec, var.tb.tv_nsec);
    log("%0*d #functions: %lu\n", var.digits, var.rank, fmap_n);
#endif // _DEBUG

    for (size_t i = 0; i < fmap_n; i++) {
        char* name;
        size_t len; // string length

        fread(&function, sizeof(size_t), 1, fp);
        fread(&len, sizeof(size_t), 1, fp);
        name = (char*) malloc(len + 1);
        fread(name, sizeof(char), len + 1, fp);
        fmap[function] = name;

        if ((int)len > max)
            max = len;
    }

    fseek(fp, sizeof(var_t), SEEK_SET);
    for (size_t i = 0; i < var.evc; i++) {
        double dt0, dt1, de0, de1;

        fread(&function, sizeof(size_t), 1, fp);
        fread(&dt0, sizeof(double), 1, fp);
        fread(&dt1, sizeof(double), 1, fp);
        fread(&de0, sizeof(double), 1, fp);
        fread(&de1, sizeof(double), 1, fp);

        log("%0*d %-*s %*.*lf %*.*lf %*.*lf %*.*lf %*.*lf %*.*lf\n", 
                var.digits, var.rank, 
                max, fmap[function],
                W, P, dt0, W, P, dt1,
                W, P, de0, W, P, de1,
                W, P, dt1 - dt0,
                W, P, de1 - de0);
    }

    fclose(fp);
}

int main(
        int argc,
        char** argv)
{
    for (int i = 1; i < argc; i++)
        process(argv[i]);

    return 0;
}

