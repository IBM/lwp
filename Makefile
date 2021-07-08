CPP  := g++
MPI  ?= /opt/ibm/spectrum_mpi
mpi  ?= mpi_ibm
CUDA := /usr/local/cuda
NVCC := nvcc
PY   := python2

define banner
	@echo
	@printf "%80s\n" | tr " " "-"
	@echo $(1)
	@printf "%80s\n" | tr " " "-"
	@echo
endef

all:	lwp.so lwp_cli bench

lwp.so:	lwp.cpp lwp_wrappers.cpp lwp_base.cpp
	$(call banner, $@)
	$(CPP) $< -o $@\
		-shared -fpic\
		-ldl\
		-I$(MPI)/include\
		-L$(MPI)/lib -l$(mpi)\
		-I$(CUDA)/include\
		-L$(CUDA)/lib64 -lcudart\
		-std=gnu++11 -Wall -Wextra

lwp_wrappers.cpp: gen.py
	$(call banner, $@)
	$(PY) $< > $@

lwp_cli:	lwp_cli.cpp lwp_base.cpp
	$(call banner, $@)
	$(CPP) $< -o $@\
		-I$(MPI)/include\
		-L$(MPI)/lib -l$(mpi)\
		-I$(CUDA)/include\
		-L$(CUDA)/lib64 -lcudart\
		-std=gnu++11 -Wall -Wextra

bench:	bench.cu
	$(call banner, $@)
	$(NVCC) $< -o $@\
		-Xcompiler -rdynamic\
		-cudart shared\
		-I$(MPI)/include\
		-L$(MPI)/lib -l$(mpi)

clean:
	$(RM) lwp.so lwp_cli lwp_wrappers.cpp bench

