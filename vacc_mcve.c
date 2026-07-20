#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <mpi.h>

#define MAX(A,B) (((A) > (B)) ? A : B)
#define ABS(a) (((a) <0) ? -(a) : (a))

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_me, world_np;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_me);
    MPI_Comm_size(MPI_COMM_WORLD, &world_np);
    if (world_me == 0) printf("standalone test_vector_acc: %d procs\n", world_np);

    const int bytes = (int)sizeof(double)*500;
    double *b;
    MPI_Win window = MPI_WIN_NULL;

    {
        MPI_Info win_info = MPI_INFO_NULL;
        MPI_Info_create(&win_info);
        MPI_Info_set(win_info, "alloc_shm", "true");
        MPI_Win_allocate((MPI_Aint)bytes, 1, win_info, MPI_COMM_WORLD,
                         &b, &window);
        MPI_Info_free(&win_info);

        MPI_Win_lock_all(MPI_MODE_NOCHECK, window);
    }
    double * const a = malloc(bytes);
    double * const c = malloc(bytes);

    for (int i = 0; i < 500; i++) {
        a[i] = i;
    }

    if(world_me==0){
        printf("--------array[%d",500);
        printf("]--------\n");
        fflush(stdout);
    }

    for (int i=0;i<500;i++) b[i]=0.;

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
            }
            MPI_Win_flush_local(0, window);
            MPI_Free_mem(contig);
        }
    }

    MPI_Win_flush_all(window);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Get_accumulate(NULL, 0, MPI_BYTE, c, bytes, MPI_BYTE, 0, 0, bytes,
                       MPI_BYTE, MPI_NO_OP, window);
    MPI_Win_flush(0, window);

    const double scale = 10.0*world_np*world_np;
    for (int i = 0; i < 500; i++) a[i] *= scale;

    for (int i = 0; i < 500; i++) {
        const double diff = a[i] - c[i];
        double max = MAX(ABS(a[i]),ABS(c[i]));
        if(max == 0. || max < .0001) max = 1.;

        if(.0001 < ABS(diff)/max){
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

    free(c);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_unlock_all(window);
    MPI_Win_free(&window);
    free(a);
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_me == 0) printf("DONE OK\n");
    MPI_Finalize();
    return 0;
}
