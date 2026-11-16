# Build:   make                 CPU programs into $(BIN) (default bin/)
#          make BIN=build/123   per-job build dir (used by the Slurm scripts)
#          make cuda            CUDA version   (needs nvcc: hpc-course-cuda)
#          make acc             OpenACC builds (needs nvc:  nvhpc)
#          make papi            prof + PAPI hardware counters
# Local (laptop) without MPI: make LOCAL_MPI_STUB=1  (single-rank stand-in)
# Test:    make test
# Flags:   make flags           (prints the exact compilers and flags)
CC     ?= cc
CFLAGS ?= -O3 -std=gnu11 -Wall -Wextra
LDLIBS ?= -lm
# OpenMP flags. macOS + Homebrew libomp:
#   make OMPFLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" OMPLIBS="-L$(brew --prefix libomp)/lib -lomp"
OMPFLAGS ?= -fopenmp
OMPLIBS  ?= -fopenmp
MPICC  ?= mpicc
NVCC   ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
NVC    ?= nvc
NVCFLAGS ?= -O3 -Minfo=accel
ACCGPU ?= -acc=gpu -gpu=cc89,cc120
PAPI_LIBS ?= -lpapi
PYTHON ?= python3
BIN    ?= bin

ifeq ($(LOCAL_MPI_STUB),1)
MPICC := $(CC)
MPIINC := -Isrc/mpi_stub
endif

# Commit id baked into every CSV row ("-dirty" if tracked files changed).
GIT_SHA := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet HEAD -- 2>/dev/null || echo -dirty)
CPPFLAGS += -DGIT_SHA=\"$(GIT_SHA)\"

COMMON := src/rng.h src/monkey.h src/cli.h src/fused.h src/kernels.h
CPU_PROGS := serial omp mpi hybrid first_match prof

.PHONY: all test flags clean cuda acc papi

all: $(addprefix $(BIN)/,$(CPU_PROGS))

$(BIN):
	mkdir -p $@

$(BIN)/serial: src/serial.c $(COMMON) | $(BIN)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BIN)/prof: src/prof.c $(COMMON) | $(BIN)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BIN)/omp: src/omp.c $(COMMON) | $(BIN)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS) $(OMPLIBS)

$(BIN)/mpi: src/mpi.c $(COMMON) | $(BIN)
	$(MPICC) $(MPIINC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BIN)/hybrid: src/mpi.c $(COMMON) | $(BIN)
	$(MPICC) $(MPIINC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS) $(OMPLIBS)

$(BIN)/first_match: src/first_match.c $(COMMON) | $(BIN)
	$(MPICC) $(MPIINC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS) $(OMPLIBS)

papi: $(BIN)/prof_papi
$(BIN)/prof_papi: src/prof.c $(COMMON) | $(BIN)
	$(CC) $(CPPFLAGS) -DUSE_PAPI $(CFLAGS) -o $@ $< $(LDLIBS) $(PAPI_LIBS)

cuda: $(BIN)/monkey_cuda
$(BIN)/monkey_cuda: cuda/monkey.cu $(COMMON) | $(BIN)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -o $@ $<

acc: $(BIN)/acc_gpu $(BIN)/acc_multicore
$(BIN)/acc_gpu: src/acc.c $(COMMON) | $(BIN)
	$(NVC) $(CPPFLAGS) -DACC_VERSION=\"acc_gpu\" $(NVCFLAGS) $(ACCGPU) -o $@ $<
$(BIN)/acc_multicore: src/acc.c $(COMMON) | $(BIN)
	$(NVC) $(CPPFLAGS) -DACC_VERSION=\"acc_multicore\" $(NVCFLAGS) -acc=multicore -o $@ $<

# Correctness: theory z-tests + exact equality of every version with serial.
# MPI programs run through $(LAUNCH) (e.g. LAUNCH="srun -n 4"); empty = 1 rank.
LAUNCH ?=
test: all
	$(PYTHON) analysis/check_theory.py --bin $(BIN)/serial
	$(PYTHON) analysis/check_theory.py --bin $(BIN)/serial --kernel v0
	$(PYTHON) analysis/check_theory.py --bin $(BIN)/serial --kernel fused
	$(PYTHON) analysis/check_repro.py --bin $(BIN)/serial
	$(PYTHON) analysis/check_repro.py --bin $(BIN)/serial --compare $(BIN)/serial --compare-args "-k fused"
	for t in 1 3 8; do OMP_NUM_THREADS=$$t $(PYTHON) analysis/check_repro.py --bin $(BIN)/serial --compare $(BIN)/omp || exit 1; done
	OMP_NUM_THREADS=5 OMP_SCHEDULE=static $(PYTHON) analysis/check_repro.py --bin $(BIN)/serial --compare $(BIN)/omp --compare-args "-k fused"
	$(PYTHON) analysis/check_repro.py --bin $(BIN)/serial --compare $(BIN)/mpi --compare-launcher "$(LAUNCH)"
	OMP_NUM_THREADS=3 $(PYTHON) analysis/check_repro.py --bin $(BIN)/serial --compare $(BIN)/hybrid --compare-launcher "$(LAUNCH)" --compare-args "-k fused"
	$(PYTHON) analysis/check_first.py --bin $(BIN)/first_match --launcher "$(LAUNCH)"

flags:
	@echo "CC=$(CC)"
	@$(CC) --version | head -1
	@echo "MPICC=$(MPICC)"
	@echo "CPPFLAGS=$(CPPFLAGS)"
	@echo "CFLAGS=$(CFLAGS)"
	@echo "LDLIBS=$(LDLIBS)"
	@echo "OMPFLAGS=$(OMPFLAGS) OMPLIBS=$(OMPLIBS)"
	@echo "NVCCFLAGS=$(NVCCFLAGS)"
	@echo "NVCFLAGS=$(NVCFLAGS) ACCGPU=$(ACCGPU)"

clean:
	rm -rf bin build
