CC     ?= cc
CFLAGS ?= -O3 -std=gnu11 -Wall -Wextra
LDLIBS ?= -lm
PYTHON ?= python3

COMMON := src/rng.h src/monkey.h

.PHONY: all test clean

all: serial

serial: src/serial.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

test: serial
	$(PYTHON) analysis/check_theory.py

clean:
	rm -f serial
