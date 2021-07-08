#!/bin/bash

nodes=2
gpn=2
len=1024
reps=1000000
streams=2
debug=""
buffer=""

function usage {
    cat << EOF
usage: ${0} [OPTIONAL]
optional parameters:
    -n, --nodes   NUM   number of nodes (default: ${nodes})
    -g, --gpn     NUM   gpus per node (default: ${gpn})
    -l, --len     NUM   array length (default: ${len})
    -r, --reps    NUM   kernel repetitions (default: ${reps})
    -s, --streams NUM   number of parallel streams per gpu (default: ${streams})
optional profiler parameters:
    -d                  debug messages
    -b, --buffer  NUM   buffer size
EOF
    exit 1
}

function fail {
    echo "[error] ${@}"
    exit 1
}

while [[ ${#} -gt 0 ]]; do
    key=${1}
    shift

    case ${key} in
        -h | --help    ) usage          ;;
        -n | --nodes   ) nodes=${1};    shift;;
        -g | --gpn     ) gpn=${1};      shift;;
        -l | --len     ) len=${1};      shift;;
        -r | --reps    ) reps=${1};     shift;;
        -s | --streams ) streams=${1};  shift;;
        -d | --debug   ) debug=1;       ;;
        -b | --buffer  ) buffer=${1};   shift;;
        * )
            fail "invalid parameter: '${key}'"
    esac
done

echo "nodes   = ${nodes}"
echo "gpn     = ${gpn}"
echo "len     = ${len}"
echo "reps    = ${reps}"
echo "streams = ${streams}"
echo "debug   = ${debug}"
echo "buffer  = ${buffer}"

cmd="
bsub
    -Is -q excl_int -W 60
    -nnodes ${nodes}
jsrun 
    --rs_per_host ${gpn}
    --gpu_per_rs 1
    -D CUDA_VISIBLE_DEVICES
    -E OMPI_LD_PRELOAD_PREPEND=./lwp.so
    -E LWP_DEBUG=${debug}
    -E LWP_MAX=${buffer}
./bench
    -p 1
    -l ${len}
    -r ${reps}
    -s ${streams}
    "

echo "${cmd}"
${cmd}    

