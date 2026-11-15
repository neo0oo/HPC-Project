# Build:   make                 (binaries go to $(BIN), default bin/)
#          make BIN=build/123   (per-job build dir, used by the Slurm scripts)
# Test:    make test
# Flags:   make flags           (prints the exact compiler and flags)
CC     ?= cc
CFLAGS ?= -O3 -std=gnu11 -Wall -Wextra
LDLIBS ?= -lm
PYTHON ?= python3
BIN    ?= bin

# Commit id baked into every CSV row ("-dirty" if tracked files changed).
GIT_SHA := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet HEAD -- 2>/dev/null || echo -dirty)
CPPFLAGS += -DGIT_SHA=\"$(GIT_SHA)\"

COMMON := src/rng.h src/monkey.h src/cli.h

.PHONY: all test flags clean

all: $(BIN)/serial

$(BIN):
	mkdir -p $@

$(BIN)/serial: src/serial.c $(COMMON) | $(BIN)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

test: $(BIN)/serial
	$(PYTHON) analysis/check_theory.py --bin $(BIN)/serial
	$(PYTHON) analysis/check_theory.py --bin $(BIN)/serial --kernel v0
	$(PYTHON) analysis/check_repro.py --bin $(BIN)/serial

flags:
	@echo "CC=$(CC)"
	@$(CC) --version | head -1
	@echo "CPPFLAGS=$(CPPFLAGS)"
	@echo "CFLAGS=$(CFLAGS)"
	@echo "LDLIBS=$(LDLIBS)"

clean:
	rm -rf bin build
