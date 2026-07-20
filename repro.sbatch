#!/bin/bash
#SBATCH --job-name=ompi-indexed-rma
#SBATCH --partition=iris
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=x86_64
#SBATCH --time=00:10:00
#SBATCH --output=repro-%j.out

set -eu

: "${MPICC:=mpicc}"
: "${MPIEXEC:=mpirun}"
: "${UCX_TLS:=rc_x,sm,self}"
: "${UCX_NET_DEVICES:=mlx5_0:1}"
export UCX_TLS UCX_NET_DEVICES

cd "${SLURM_SUBMIT_DIR}"

indexed="vacc-indexed-${SLURM_JOB_ID}"
predefined="vacc-mpi-double-${SLURM_JOB_ID}"

"${MPICC}" -std=c99 -O2 vacc_mcve.c -o "${indexed}" -lm
"${MPICC}" -DUSE_MPI_DOUBLE -std=c99 -O2 vacc_mcve.c \
    -o "${predefined}" -lm

echo "Open MPI:"
"${MPIEXEC}" --version | head -2
echo "UCX_TLS=${UCX_TLS}"
echo "UCX_NET_DEVICES=${UCX_NET_DEVICES}"

run_case()
{
    name=$1
    binary=$2
    ok=0
    corrupt=0
    other=0

    for run in 1 2 3; do
        output="${name}-${SLURM_JOB_ID}-${run}.log"
        set +e
        timeout -k 5 55 "${MPIEXEC}" --timeout 45 -n "${SLURM_NTASKS}" \
            --map-by ppr:2:node -x UCX_TLS -x UCX_NET_DEVICES \
            "./${binary}" >"${output}" 2>&1
        status=$?
        set -e

        if grep -qa 'DONE OK' "${output}"; then
            ok=$((ok + 1))
            echo "${name} run ${run}: OK"
        elif grep -qaE 'ERROR:|Bailing out' "${output}"; then
            corrupt=$((corrupt + 1))
            first_error=$(grep -m1 -a 'ERROR:' "${output}" | tr -d '\000')
            echo "${name} run ${run}: CORRUPT ${first_error}"
        else
            other=$((other + 1))
            echo "${name} run ${run}: other (status=${status})"
            tail -10 "${output}"
        fi
    done

    echo "${name}: OK=${ok} CORRUPT=${corrupt} other=${other}"
}

run_case indexed "${indexed}"
run_case mpi-double "${predefined}"
