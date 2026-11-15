#!/usr/bin/env python3
"""Infinite-monkey theory: the formulas every measurement is compared with.

Alphabet: 27 symbols (a-z and space), each typed with probability 1/27.

expected_count(n, L)            overlapping occurrences in n characters
expected_count_chunked(n, C, L) same, typed as chunks of C (boundaries not counted)
expected_wait(target)           mean characters until the first occurrence
                                (Conway's overlap formula)

As a script, prints how long targets take at a given speed, e.g.
    python3 analysis/theory.py --rate 3e8 --cores 64 romeo wherefore
"""
from __future__ import annotations

import argparse

K = 27


def canonical(target: str) -> str:
    """Lower-case, '_' -> ' ', validate alphabet."""
    t = target.lower().replace("_", " ")
    bad = [c for c in t if not (c == " " or "a" <= c <= "z")]
    if bad or not t:
        raise ValueError(f"invalid target {target!r}")
    return t


def expected_count(n: int, length: int) -> float:
    return max(n - length + 1, 0) / K**length


def expected_count_chunked(n: int, chunk: int, length: int) -> float:
    full, rest = divmod(n, chunk)
    return full * expected_count(chunk, length) + expected_count(rest, length)


def expected_wait(target: str) -> int:
    """Mean number of characters typed until `target` first appears."""
    t = canonical(target)
    L = len(t)
    return sum(K**j for j in range(1, L + 1) if t[:j] == t[L - j:])


def human_time(seconds: float) -> str:
    for unit, s in (("y", 31_557_600), ("d", 86_400), ("h", 3_600), ("min", 60)):
        if seconds >= s:
            return f"{seconds / s:.1f} {unit}"
    return f"{seconds:.2f} s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*",
                    default=["a", "the", "love", "romeo", "juliet", "capulet",
                             "montague", "wherefore"])
    ap.add_argument("--rate", type=float, default=3e8,
                    help="characters per second per core (default 3e8)")
    ap.add_argument("--cores", type=int, default=1)
    args = ap.parse_args()

    total_rate = args.rate * args.cores
    print(f"rate {args.rate:.3g} chars/s/core x {args.cores} cores")
    for target in args.targets:
        t = canonical(target)
        wait = expected_wait(t)
        print(f"  {t!r:>14}  L={len(t):<2}  mean wait {wait:.3e} chars"
              f"  ~ {human_time(wait / total_rate)}")


if __name__ == "__main__":
    main()
