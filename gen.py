#
#   IBM Corporation (C) 2019
#   Nelson Mimura -- nmimura@ibm.com
#

'''
    ('int',
     'MPI_Barrier',
        ('MPI_Comm', 'comm')),
'''

# A
# ------------------------------------------------------------------------------

fs_A = [

    ('int',
     'MPI_Init',
        ('int*', 'argc'),
        ('char***', 'argv')),

]

# B
# ------------------------------------------------------------------------------

fs_B = [
    
    ('cudaError_t',
     'cudaLaunch',
        ('const void*', 'func')),

    ('cudaError_t',
     'cudaLaunchKernel',
        ('const void*', 'func'),
        ('dim3', 'gridDim'),
        ('dim3', 'blockDim'),
        ('void**', 'args'),
        ('size_t', 'sharedMem'),
        ('cudaStream_t', 'stream')),

]

# ------------------------------------------------------------------------------

fs_C = [

    ('cudaError_t',
     'cudaMalloc',
        ('void**', 'devPtr'),
        ('size_t', 'size')),

    ('cudaError_t',
     'cudaMallocHost',
        ('void**', 'devPtr'),
        ('size_t', 'size')),

    ('cudaError_t',
     'cudaMemcpy',
        ('void*', 'dst'),
        ('const void*', 'src'),
        ('size_t', 'count'),
        ('enum cudaMemcpyKind', 'kind')),

]

# ------------------------------------------------------------------------------

fs = fs_A + fs_B + fs_C

# ------------------------------------------------------------------------------

pre = {
}

pos = {

    'MPI_Init' : [
        '\tcudaDeviceSynchronize();',
        '\tMPI_Barrier(MPI_COMM_WORLD);',
        '\tcudaEventCreate(&var.eb);',
        '\tcudaEventRecord(var.eb);',
        '\tcudaEventSynchronize(var.eb);',
        '\tclock_gettime(CLOCK_MONOTONIC, &var.tb);',
    ],

}

TYPE = 0
NAME = 1
ARGS = 2

# function strings
print 'const char* __fs[] = {'
for f in fs:
    print '\t"%s",' % (f[NAME])
print '};'
print

for f in fs:

    mpi = f[NAME].startswith('MPI_')
    cudart = f[NAME].startswith('cuda')

    # comment
    print '//'
    print '//\t%s' % (f[NAME])
    print '//\t', len(f[NAME]) * '-'
    print '//'
    print

    # typedef
    print 'typedef',
    print f[TYPE],
    print '(*__ft_%s)' %(f[NAME]),
    print '('
    args = []
    for a in f[ARGS:]:
        args.append('\t%s %s' % (a[TYPE], a[NAME]))
    print ',\n'.join(args)
    print ');'
    print

    # declare function pointer
    print '__ft_%s __fp_%s;' % (f[NAME], f[NAME])
    print

    # initialize function pointer
    print '__attribute__((constructor)) void __fc_%s ()' % (f[NAME])
    print '{'
    print '\t__fp_%s = (__ft_%s)\n\t\tdlsym(RTLD_NEXT, "%s");' % (\
            f[NAME], f[NAME], f[NAME])
    print '}'
    print

    # wrapper
    print '%s %s (' % (f[TYPE], f[NAME])
    args = []
    for a in f[ARGS:]:
        args.append('\t%s %s' % (a[TYPE], a[NAME]))
    print ',\n'.join(args),
    print ')'
    print '{'

    # function
    if 'cudaLaunch' in f[NAME]:
        function = 'func'
    else:
        function = '__fp_' + f[NAME]

    # stream
    if 'stream' in (x[NAME] for x in f[ARGS:]):
        stream = 'stream'
    else:
        stream = '0'

    # wrapper start
    print '\tif (!lwp_initialized)'
    print '\t\tlwp_init();'
    print
    print '\tev_t* ev = &var.evs[var.evc++];'
    print '\tev->function = (size_t) %s;' % (function)
    print '\tev->stream = %s;' % (stream)
    print

    # wrapper pre

    # wrapper start timers
    if cudart:
        print '\tcudaEventRecord(ev->e0, %s);' % (stream)
    print '\tclock_gettime(CLOCK_MONOTONIC, &ev->t0);'
    print

    # wrapper call
    print '\t%s rc = __fp_%s' %(f[TYPE], f[NAME]),
    print '('
    args = []
    for a in f[ARGS:]:
        args.append('\t\t\t%s' % (a[NAME]))
    print ',\n'.join(args),
    print ');'
    print

    # wrapper end timers
    print '\tclock_gettime(CLOCK_MONOTONIC, &ev->t1);'
    if cudart:
        print '\tcudaEventRecord(ev->e1, %s);' % (stream)
    print

    # wrapper pos
    if f[NAME] in pos:
        print '\n'.join(pos[f[NAME]])
        print

    # wrapper return
    print '\treturn rc;'

    # wrapper end
    print '}'
    print
