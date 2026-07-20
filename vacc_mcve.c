#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>

#define MAX(A,B) (((A) > (B)) ? A : B)
#define ABS(a) (((a) <0) ? -(a) : (a))

typedef struct gmr_s {
    void **slices;
} gmr_t;

gmr_t *mreg = NULL;
MPI_Win window = MPI_WIN_NULL;

#define MAXPROC 128
int world_me, world_np;

static void GetPermutedProcList(int *ProcList)
{
    for (int i=0; i< world_np; i++) ProcList[i]=i;
    if(world_np ==1) return;

    (void)srand((unsigned)world_me);

    for (int i=0; i< world_np; i++){
        const int iswap = (int)(rand() % world_np);
        const int temp = ProcList[iswap];
        ProcList[iswap] = ProcList[i];
        ProcList[i] = temp;
    }
}

static void test_vector_acc(void)
{
    const int bytes = (int)sizeof(double)*500;
    double *b[MAXPROC];

    {
        mreg = malloc(sizeof(gmr_t));
        mreg->slices = malloc(sizeof(void *)*world_np);
        void ** const alloc_slices = malloc(sizeof(void *)*world_np);

        MPI_Info win_info = MPI_INFO_NULL;
        MPI_Info_create(&win_info);
        MPI_Info_set(win_info, "alloc_shm", "true");
        MPI_Win_allocate((MPI_Aint)bytes, 1, win_info, MPI_COMM_WORLD, &alloc_slices[world_me], &window);
        MPI_Info_free(&win_info);
        MPI_Allgather(&alloc_slices[world_me], sizeof(void *), MPI_BYTE, alloc_slices, sizeof(void *), MPI_BYTE, MPI_COMM_WORLD);

        for (int gi = 0; gi < world_np; gi++) {
            b[gi] = alloc_slices[gi];
        }

        for (int gi = 0; gi < world_np; gi++) {
            mreg->slices[gi] = alloc_slices[gi];
        }
        free(alloc_slices);

        MPI_Win_lock_all(MPI_MODE_NOCHECK, window);
    }
    double * const a = malloc(bytes);
    double * const c = malloc(bytes);

    for (int i = 0; i < 500; i++) {
        a[i] = i % 500;
    }

    if(world_me==0){
        printf("--------array[%d",500);
        printf("]--------\n");
        fflush(stdout);
    }

    int proclist[MAXPROC];
    GetPermutedProcList(proclist);

    for (int i=0;i<500;i++) b[world_me][i]=0.;

    sleep(1);

    void *src_ptr_array[250];
    void *dst_ptr_array[250];
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0;i<100*world_np;i++){
        for (int j=0; j<250; j++){
            src_ptr_array[j]= 2*j + a;
            dst_ptr_array[j]= 2*j + b[0];
        }
        {
            {
                void ** const src_buf = malloc(251*sizeof(void*));
                char *contig;
                MPI_Alloc_mem((MPI_Aint)250*sizeof(double), MPI_INFO_NULL, &contig);
                src_buf[250] = contig;

                for (int i = 0; i < 250; i++) {
                    src_buf[i] = contig + (MPI_Aint)i*sizeof(double);
                    {
                        double * const s_in = (double*) src_ptr_array[i];
                        double * const s_out = (double*) src_buf[i];
                        s_out[0] = s_in[0] * 0.1;
                    }
                }

                {
                    MPI_Datatype  type_loc, type_rem;
                    int           disp_loc[250];
                    int           disp_rem[250];
                    MPI_Aint      loc_addr[250];
                    MPI_Aint      base_loc;
                    MPI_Aint      base_rem;

                    MPI_Get_address(mreg->slices[0], &base_rem);

                    void *base_loc_ptr = src_buf[0];
                    MPI_Get_address(src_buf[0], &base_loc);
                    for (int i = 0; i < 250; i++) {
                        MPI_Get_address(src_buf[i], &loc_addr[i]);
                        if (loc_addr[i] < base_loc) { base_loc = loc_addr[i]; base_loc_ptr = src_buf[i]; }
                    }

                    for (int i = 0; i < 250; i++) {
                        MPI_Aint target_rem;
                        MPI_Get_address(dst_ptr_array[i], &target_rem);
                        disp_rem[i]  = (int)((target_rem - base_rem)/(MPI_Aint)sizeof(double));

                        disp_loc[i]  = (int)((loc_addr[i] - base_loc)/(MPI_Aint)sizeof(double));
                    }

                    for (int start = 0; start < 250; start++) {
                        MPI_Type_create_indexed_block(1, 1, &disp_loc[start], MPI_DOUBLE, &type_loc);
                        MPI_Type_create_indexed_block(1, 1, &disp_rem[start], MPI_DOUBLE, &type_rem);
                        MPI_Type_commit(&type_loc);
                        MPI_Type_commit(&type_rem);

                        MPI_Accumulate(base_loc_ptr, 1, type_loc, 0, 0, 1, type_rem, MPI_SUM, window);

                        MPI_Type_free(&type_loc);
                        MPI_Type_free(&type_rem);
                    }
                    MPI_Win_flush_local(0, window);
                }

                MPI_Free_mem(src_buf[250]);
                free(src_buf);
            }
        }
        for (int j=0; j<250; j++){
            src_ptr_array[j]= 2*j+1 + a;
            dst_ptr_array[j]= 2*j+1 + b[0];
        }
        {
            {
                void ** const src_buf = malloc(251*sizeof(void*));
                char *contig;
                MPI_Alloc_mem((MPI_Aint)250*sizeof(double), MPI_INFO_NULL, &contig);
                src_buf[250] = contig;

                for (int i = 0; i < 250; i++) {
                    src_buf[i] = contig + (MPI_Aint)i*sizeof(double);
                    {
                        double * const s_in = (double*) src_ptr_array[i];
                        double * const s_out = (double*) src_buf[i];
                        s_out[0] = s_in[0] * 0.1;
                    }
                }

                {
                    MPI_Datatype  type_loc, type_rem;
                    int           disp_loc[250];
                    int           disp_rem[250];
                    MPI_Aint      loc_addr[250];
                    MPI_Aint      base_loc;
                    MPI_Aint      base_rem;

                    MPI_Get_address(mreg->slices[0], &base_rem);

                    void *base_loc_ptr = src_buf[0];
                    MPI_Get_address(src_buf[0], &base_loc);
                    for (int i = 0; i < 250; i++) {
                        MPI_Get_address(src_buf[i], &loc_addr[i]);
                        if (loc_addr[i] < base_loc) { base_loc = loc_addr[i]; base_loc_ptr = src_buf[i]; }
                    }

                    for (int i = 0; i < 250; i++) {
                        MPI_Aint target_rem;
                        MPI_Get_address(dst_ptr_array[i], &target_rem);
                        disp_rem[i]  = (int)((target_rem - base_rem)/(MPI_Aint)sizeof(double));

                        disp_loc[i]  = (int)((loc_addr[i] - base_loc)/(MPI_Aint)sizeof(double));
                    }

                    for (int start = 0; start < 250; start++) {
                        MPI_Type_create_indexed_block(1, 1, &disp_loc[start], MPI_DOUBLE, &type_loc);
                        MPI_Type_create_indexed_block(1, 1, &disp_rem[start], MPI_DOUBLE, &type_rem);
                        MPI_Type_commit(&type_loc);
                        MPI_Type_commit(&type_rem);

                        MPI_Accumulate(base_loc_ptr, 1, type_loc, 0, 0, 1, type_rem, MPI_SUM, window);

                        MPI_Type_free(&type_loc);
                        MPI_Type_free(&type_rem);
                    }
                    MPI_Win_flush_local(0, window);
                }

                MPI_Free_mem(src_buf[250]);
                free(src_buf);
            }
        }
    }

    MPI_Win_flush_all(window);
    MPI_Barrier(MPI_COMM_WORLD);

    {
        const MPI_Aint disp = (MPI_Aint)((uint8_t*)b[0] -
                                        (uint8_t*)mreg->slices[0]);
        MPI_Get_accumulate(NULL, 0, MPI_BYTE, c, bytes, MPI_BYTE, 0,
                           disp, bytes, MPI_BYTE, MPI_NO_OP, window);
        MPI_Win_flush(0, window);
    }

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
    free(mreg->slices);
    free(mreg);
    free(a);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_me);
    MPI_Comm_size(MPI_COMM_WORLD, &world_np);
    if (world_me == 0) printf("standalone test_vector_acc: %d procs\n", world_np);
    if(world_np > MAXPROC) {
        fprintf(stderr, "nproc to big (max=%d)\n", MAXPROC);
        MPI_Abort(MPI_COMM_WORLD, world_np);
    }
    test_vector_acc();
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_me == 0) printf("DONE OK\n");
    MPI_Finalize();
    return 0;
}
