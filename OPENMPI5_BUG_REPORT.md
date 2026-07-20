# `MPI_Accumulate` loses updates with a one-element indexed datatype over UCX/InfiniBand

## Summary

Open MPI 5.0.10rc2 loses `MPI_Accumulate` updates when the
origin and target datatypes are committed indexed datatypes, even though each
datatype has the type signature of exactly one `MPI_DOUBLE`.

The reproducer below fails deterministically in the tested configuration: four
ranks on two nodes using the UCX PML over InfiniBand.  One element in rank 0's
window remains zero instead of containing the accumulated value.

Changing only the datatype representation from
`MPI_Type_create_indexed_block(1, 1, ..., MPI_DOUBLE, ...)` to the predefined
`MPI_DOUBLE` makes the test pass.  The logical origin and target addresses,
element counts, operation, window, synchronization, and transport are
otherwise unchanged.

## Environment

- Open MPI: 5.0.10rc2
- Language: C99
- Processes: 4, with 2 processes per node on 2 nodes
- Architecture: x86_64
- PML/transport: UCX over InfiniBand
- `UCX_TLS=rc_x,sm,self`
- `UCX_NET_DEVICES=mlx5_0:1`
- Window: `MPI_Win_allocate`, one 500-double window per rank
- Epoch: passive-target `MPI_Win_lock_all`

The result was reproduced on three different node types:

| Node type | Indexed datatypes | Predefined `MPI_DOUBLE` control |
| --- | --- | --- |
| Iris | `CORRUPT` in 3/3 runs | `OK` in 3/3 runs |
| Thor | `CORRUPT` in 3/3 runs | `OK` in 3/3 runs |
| Rome | `CORRUPT` in 3/3 runs | `OK` in 3/3 runs |

The Rome test used a UCX build compiled against the node's rdma-core ABI.  A
TCP login-node control passes with either datatype representation.

## Expected behavior

All 500 elements in rank 0's window should contain the sum of all accumulate
operations.  The program should print:

```text
 OK

DONE OK
```

The indexed datatypes each contain one block of one `MPI_DOUBLE`, so their type
signatures match the direct `MPI_DOUBLE` operations in the control build.

## Actual behavior

The indexed-datatype build consistently reports that the third element remains
zero.  Representative output is:

```text
standalone test_vector_acc: 4 procs
--------array[500]--------
ERROR: a [3] (proc=0):320.000000 b [3] 0.000000

A = 320.000000 B = 0.000000
[0] ARMCI Error: Bailing out
```

The rank reporting the mismatch can vary with the node type, but the missing
value is reproducible.  The `USE_MPI_DOUBLE` build completes successfully.

## Reproducer

```c
#include <stdio.h>
#include <math.h>
#include <unistd.h>

#include <mpi.h>

#define MAX(A,B) (((A) > (B)) ? A : B)

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_me, world_np;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_me);
    MPI_Comm_size(MPI_COMM_WORLD, &world_np);
    if (world_me == 0) printf("standalone test_vector_acc: %d procs\n", world_np);

    const MPI_Aint bytes = (MPI_Aint)sizeof(double)*500;
    static double a[500], c[500];
    double *b;
    MPI_Win window = MPI_WIN_NULL;

    MPI_Win_allocate(bytes, 1, MPI_INFO_NULL, MPI_COMM_WORLD,
                     &b, &window);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, window);

    for (int i = 0; i < 500; i++) {
        a[i] = i;
        b[i] = 0.;
    }

    if(world_me==0){
        printf("--------array[%d",500);
        printf("]--------\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0;i<100*world_np;i++){
        for (int parity = 0; parity < 2; parity++) {
            double *contig;
            MPI_Alloc_mem((MPI_Aint)250*sizeof(double), MPI_INFO_NULL,
                          &contig);

            for (int j = 0; j < 250; j++) {
                contig[j] = a[2*j+parity] * 0.1;
            }

            for (int start = 0; start < 250; start++) {
#ifndef USE_MPI_DOUBLE
                const int disp_loc = start;
                const int disp_rem = 2*start+parity;
                MPI_Datatype type_loc, type_rem;
                MPI_Type_create_indexed_block(1, 1, &disp_loc, MPI_DOUBLE,
                                              &type_loc);
                MPI_Type_create_indexed_block(1, 1, &disp_rem, MPI_DOUBLE,
                                              &type_rem);
                MPI_Type_commit(&type_loc);
                MPI_Type_commit(&type_rem);

                MPI_Accumulate(contig, 1, type_loc, 0, 0, 1, type_rem,
                               MPI_SUM, window);

                MPI_Type_free(&type_loc);
                MPI_Type_free(&type_rem);
#else
                const MPI_Aint target_disp =
                    (MPI_Aint)(2*start+parity)*sizeof(double);
                MPI_Accumulate(&contig[start], 1, MPI_DOUBLE, 0,
                               target_disp, 1, MPI_DOUBLE, MPI_SUM, window);
#endif
            }
            MPI_Win_flush_local(0, window);
            MPI_Free_mem(contig);
        }
    }

    MPI_Win_flush_all(window);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Get(c, (int)bytes, MPI_BYTE, 0, 0, (int)bytes, MPI_BYTE, window);
    MPI_Win_flush(0, window);

    const double scale = 10.0*world_np*world_np;
    for (int i = 0; i < 500; i++) a[i] *= scale;

    for (int i = 0; i < 500; i++) {
        const double diff = a[i] - c[i];
        double max = MAX(fabs(a[i]),fabs(c[i]));
        if(max == 0. || max < .0001) max = 1.;

        if(.0001 < fabs(diff)/max){
            char msg[48];
            sprintf(msg,"(proc=%d):%f",world_me,a[i]);
            printf("ERROR: a [%d] %s", i+1, msg);
            sprintf(msg,"%f\n",c[i]);
            printf(" b [%d] %s", i+1, msg);
            printf("\nA = %f B = %f\n", a[i], c[i]);
            fflush(stdout);
            sleep(1);
            fprintf(stderr, "[%d] ARMCI Error: Bailing out\n", world_me);
            fflush(NULL);
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(0==world_me){
        printf(" OK\n\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_unlock_all(window);
    MPI_Win_free(&window);
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_me == 0) printf("DONE OK\n");
    MPI_Finalize();
    return 0;
}
```

## Build and run

Failing indexed-datatype build:

```sh
mpicc -std=c99 -O2 -Wall -Wextra vacc_mcve.c -o vacc-indexed -lm
export UCX_TLS=rc_x,sm,self
export UCX_NET_DEVICES=mlx5_0:1
mpirun -n 4 --map-by ppr:2:node \
    -x UCX_TLS -x UCX_NET_DEVICES ./vacc-indexed
```

Working predefined-datatype control:

```sh
mpicc -DUSE_MPI_DOUBLE -std=c99 -O2 -Wall -Wextra \
    vacc_mcve.c -o vacc-mpi-double -lm
mpirun -n 4 --map-by ppr:2:node \
    -x UCX_TLS -x UCX_NET_DEVICES ./vacc-mpi-double
```

## A/B comparison

The failing operation creates an origin datatype whose sole element is
`contig[start]` and a target datatype whose sole element is
`b[2*start+parity]`:

```c
MPI_Type_create_indexed_block(1, 1, &disp_loc, MPI_DOUBLE, &type_loc);
MPI_Type_create_indexed_block(1, 1, &disp_rem, MPI_DOUBLE, &type_rem);
MPI_Accumulate(contig, 1, type_loc, 0, 0, 1, type_rem,
               MPI_SUM, window);
```

The working branch expresses the same operation directly:

```c
MPI_Accumulate(&contig[start], 1, MPI_DOUBLE, 0,
               (MPI_Aint)(2*start+parity)*sizeof(double),
               1, MPI_DOUBLE, MPI_SUM, window);
```

The window displacement unit is one byte.  Displacements in an indexed
datatype are expressed in units of the old datatype's extent, so both forms
select the same doubles.

## Additional isolation results

- Replacing the final `MPI_Get_accumulate(..., MPI_NO_OP, ...)` readback from
  the original application with `MPI_Get` does not change the failure.
- Setting the `alloc_shm` window info key to false does not change the failure;
  the reproducer now uses `MPI_INFO_NULL`.
- Replacing the indexed datatypes with `MPI_DOUBLE` is sufficient to make all
  tested UCX/InfiniBand runs pass.
- The indexed build passes in the tested TCP control.
- The reproducer compiles cleanly with `-std=c99 -O2 -Wall -Wextra`.
