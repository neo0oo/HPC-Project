# Source this file (do not execute it): `source env/modules.sh`
# Loads the exact toolchain used for every experiment.
#
# OpenMPI/4.1.5-GCC-12.3.0 also brings GCC 12.3.0 (EasyBuild foss-2023a
# toolchain). Gotcha seen on this cluster: a system /usr/bin/mpicc can shadow
# the module's mpicc, so we resolve the compilers explicitly after loading.

if ! type module >/dev/null 2>&1; then
    # Non-interactive shells may not have the module function yet.
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
             /usr/share/lmod/lmod/init/bash; do
        [ -r "$f" ] && . "$f" && break
    done
fi

module purge >/dev/null 2>&1 || true
module load OpenMPI/4.1.5-GCC-12.3.0

export CC=gcc
export MPICC="$(dirname "$(command -v mpirun)")/mpicc"

# Determinism / clean measurements.
export LC_ALL=C
