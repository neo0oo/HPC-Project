# Build:   make                 (binaries go to $(BIN), default bin/)
#          make BIN=build/123   (per-job build dir, used by the Slurm scripts)
# Test:    make test
# Flags:   make flags           (prints the exact compiler and flags)
CC     ?= cc
CFLAGS ?= -O3 -std=gnu11 -Wall -Wextra
LDLIBS ?= -lm
# OpenMP flags. macOS + Homebrew libomp:
#   make OMPFLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" OMPLIBS="-L$(brew --prefix libomp)/lib -lomp"
OMPFLAGS ?= -fopenmp
OMPLIBS  ?= -fopenmp
PYTHON ?= python3
BIN    ?= bin

# Commit id baked into every CSV row ("-dirty" if tracked files changed).
GIT_SHA := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet HEAD -- 2>/dev/null || echo -dirty)
CPPFLAGS += -DGIT_SHA=\"$(GIT_SHA)\"

COMMON := src/rng.h src/monkey.h src/cli.h

.PHONY: all test flags clean

all: $(BIN)/serial $(BIN)/omp

$(BIN):
	mkdir -p $@

$(BIN)/serial: src/serial.c $(COMMON) | $(BIN)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BIN)/omp: src/omp.c $(COMMON) | $(BIN)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS) $(OMPLIBS)

test: all
	$(PYTHON) analysis/check_theory.py --bin $(BIN)/serial
	$(PYTHON) analysis/check_theory.py --bin $(BIN)/serial --kernel v0
	$(PYTHON) analysis/check_repro.py --bin $(BIN)/serial
	OMP_NUM_THREADS=4 $(PYTHON) analysis/check_theory.py --bin $(BIN)/omp
	for t in 1 3 8; do OMP_NUM_THREADS=$$t $(PYTHON) analysis/check_repro.py --bin $(BIN)/serial --compare $(BIN)/omp || exit 1; done
	OMP_NUM_THREADS=5 OMP_SCHEDULE=static $(PYTHON) analysis/check_repro.py --bin $(BIN)/serial --compare $(BIN)/omp

flags:
	@echo "CC=$(CC)"
	@$(CC) --version | head -1
	@echo "CPPFLAGS=$(CPPFLAGS)"
	@echo "CFLAGS=$(CFLAGS)"
	@echo "LDLIBS=$(LDLIBS)"
	@echo "OMPFLAGS=$(OMPFLAGS) OMPLIBS=$(OMPLIBS)"

clean:
	rm -rf bin build
