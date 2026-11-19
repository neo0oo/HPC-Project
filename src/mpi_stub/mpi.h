/*
 * Minimal single-process stand-in for MPI, ONLY for compiling and testing the
 * MPI programs on a laptop without an MPI installation:
 *     make LOCAL_MPI_STUB=1
 * It implements exactly the calls our programs use, for one rank. Every real
 * experiment uses OpenMPI on the cluster.
 */
#ifndef MPI_STUB_H
#define MPI_STUB_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Request;
typedef struct { int dummy; } MPI_Status;

#define MPI_COMM_WORLD 0
#define MPI_SUCCESS 0
#define MPI_THREAD_FUNNELED 1
#define MPI_IN_PLACE ((void *)1)
#define MPI_STATUS_IGNORE ((MPI_Status *)0)
#define MPI_REQUEST_NULL 0

enum { MPI_UINT64_T = 8, MPI_DOUBLE = 9, MPI_INT = 4, MPI_CHAR = 1 };
enum { MPI_SUM, MPI_MIN, MPI_MAX };

static inline size_t mpi_stub_size(MPI_Datatype t)
{
    return t == MPI_UINT64_T ? 8 : t == MPI_DOUBLE ? 8 : t == MPI_INT ? 4 : 1;
}
static inline int MPI_Init_thread(int *a, char ***b, int req, int *prov)
{ (void)a; (void)b; *prov = req; return 0; }
static inline int MPI_Finalize(void) { return 0; }
static inline int MPI_Comm_rank(MPI_Comm c, int *r) { (void)c; *r = 0; return 0; }
static inline int MPI_Comm_size(MPI_Comm c, int *s) { (void)c; *s = 1; return 0; }
static inline int MPI_Barrier(MPI_Comm c) { (void)c; return 0; }
static inline int MPI_Abort(MPI_Comm c, int e) { (void)c; exit(e); }
static inline double MPI_Wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static inline int MPI_Reduce(const void *s, void *r, int n, MPI_Datatype t,
                             MPI_Op op, int root, MPI_Comm c)
{
    (void)op; (void)root; (void)c;
    if (s != MPI_IN_PLACE)
        memcpy(r, s, n * mpi_stub_size(t));
    return 0;
}
static inline int MPI_Allreduce(const void *s, void *r, int n, MPI_Datatype t,
                                MPI_Op op, MPI_Comm c)
{ return MPI_Reduce(s, r, n, t, op, 0, c); }
static inline int MPI_Iallreduce(const void *s, void *r, int n, MPI_Datatype t,
                                 MPI_Op op, MPI_Comm c, MPI_Request *q)
{ *q = 1; return MPI_Reduce(s, r, n, t, op, 0, c); }
static inline int MPI_Wait(MPI_Request *q, MPI_Status *st)
{ (void)st; *q = MPI_REQUEST_NULL; return 0; }
static inline int MPI_Bcast(void *b, int n, MPI_Datatype t, int root, MPI_Comm c)
{ (void)b; (void)n; (void)t; (void)root; (void)c; return 0; }
static inline int MPI_Get_processor_name(char *name, int *len)
{ strcpy(name, "stub"); *len = 4; return 0; }

#endif
