# Open MPI 5 indexed-datatype RMA reproducer

This directory contains a minimal C reproducer for an Open MPI 5 RMA data
corruption observed with UCX over InfiniBand.  The reproducer is correct with
the TCP transport, but on two InfiniBand-connected nodes an element in the
target window can remain zero after repeated `MPI_Accumulate` operations.

The isolated trigger is the datatype representation.  The default build uses
`MPI_Type_create_indexed_block` to describe a single `MPI_DOUBLE`.  Defining
`USE_MPI_DOUBLE` performs the same transfers with the predefined
`MPI_DOUBLE` datatype and works correctly in the tested configuration.

## Observed behavior

The tests used Open MPI 5.0.10rc2 with four ranks.  The two-node failing cases
used two ranks per node, UCX transports `rc_x,sm,self`, and InfiniBand device
`mlx5_0:1`.  The same result was observed on Iris, Thor, and Rome x86_64
nodes.

| Build | Transport | Result |
| --- | --- | --- |
| default indexed datatype | Iris UCX/InfiniBand, two nodes | `CORRUPT` in 3/3 runs |
| default indexed datatype | Thor UCX/InfiniBand, two nodes | `CORRUPT` in 3/3 runs |
| default indexed datatype | Rome UCX/InfiniBand, two nodes | `CORRUPT` in 3/3 runs |
| `USE_MPI_DOUBLE` | Iris, Thor, and Rome UCX/InfiniBand | `OK` in 3/3 runs on each |
| either build | TCP login-node control | `OK` |

Representative output is preserved in
[`logs/indexed-corrupt.log`](logs/indexed-corrupt.log) and
[`logs/mpi-double-ok.log`](logs/mpi-double-ok).

## Build and run

Build the reproducing indexed-datatype case:

```sh
mpicc -std=c99 -O2 -Wall -Wextra vacc_mcve.c -o vacc-indexed -lm
mpirun -n 4 ./vacc-indexed
```

Build the predefined-datatype control:

```sh
mpicc -DUSE_MPI_DOUBLE -std=c99 -O2 -Wall -Wextra \
    vacc_mcve.c -o vacc-mpi-double -lm
mpirun -n 4 ./vacc-mpi-double
```

For the tested two-node Slurm configuration, activate an Open MPI 5 build
with UCX support and submit:

```sh
sbatch --export=ALL repro.sbatch
```

The batch script contains no user-specific paths.  It expects `mpicc` and
`mpirun` to resolve to the Open MPI installation being tested.  `UCX_TLS` and
`UCX_NET_DEVICES` may be set before submission; otherwise the tested values
are used as defaults.

The Open MPI build used for Iris and Thor was linked against an Iris-native
UCX installation.  On Rome, that UCX build could not expose the verbs device,
so the test used a UCX build compiled against Rome's rdma-core ABI and
preloaded it into the same Open MPI 5 installation.  Confirm with `ucx_info
-d` that `rc_mlx5` or `rc_verbs` and `mlx5_0:1` are available before treating
a Rome result as an InfiniBand test.

## What differs between the failing and working paths

Each logical transfer contains one double.  In the failing default path, the
origin and target are represented by committed indexed datatypes containing
one block of length one.  A new pair of datatypes is created, committed, used
by `MPI_Accumulate`, and freed for every element.

The `USE_MPI_DOUBLE` branch addresses the same origin and target elements but
uses `MPI_DOUBLE` directly.  Since the window displacement unit is one byte,
the target element index is converted to a byte displacement in that branch.
This branch did not reproduce the corruption.

Changing the result readback from `MPI_Get_accumulate` with `MPI_NO_OP` to
`MPI_Get` did not affect the bug.  The `alloc_shm` window info hint also made
no difference, and the final reproducer uses `MPI_INFO_NULL`.

## Reduction history

The Git history records the complete reduction rather than only the final
source.  The work was collaborative: early coverage-guided and mechanical
reduction was performed with Claude alongside manual edits, and later
constant-propagation and controlled Iris experiments were performed with
Codex alongside additional manual edits.

Major milestones are:

- `e6b243e`: establish a 3,350-line pure-MPI baseline reproducer.
- `49a1a4d` through `c078164`: remove unused ARMCI operations, declarations,
  macros, and uncovered branches using compiler and coverage evidence.
- `b1d6926` through `b71fa68`: inline the remaining ARMCI buffer, GMR, RMA,
  initialization, and finalization layers into MPI operations.
- `f20ca5a` through `01d4b96`: propagate fixed environment, communicator,
  global-state, registration, and window choices.
- `772d8a2` through `e3de90d`: reduce the active test with dead-code removal,
  C99 declarations, fixed dimensions, direct displacements, and helper
  inlining.
- `85930d8` and `d334683`: show that `alloc_shm` and
  `MPI_Get_accumulate` readback are not required.
- `15fd5b5`: add the predefined-`MPI_DOUBLE` comparison and demonstrate that
  the one-double indexed datatype is the distinguishing trigger.

The per-commit messages contain the corresponding compiler, TCP-control, and
two-node UCX/InfiniBand verification results.
