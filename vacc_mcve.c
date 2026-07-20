#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>

#define MAX(A,B) (((A) > (B)) ? A : B)
#define ABS(a) (((a) <0) ? -(a) : (a))

typedef struct {
    void **src_ptr_array;
    void **dst_ptr_array;
    int    bytes;
    int    ptr_array_len;
} armci_giov_t;

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

    const int proc = 0;
    const double alpha = 0.1;
    armci_giov_t dsc = {
        .src_ptr_array = (void *[250]){0},
        .dst_ptr_array = (void *[250]){0},
        .bytes = sizeof(double),
        .ptr_array_len = 250
    };
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0;i<100*world_np;i++){
        for (int j=0; j<250; j++){
            dsc.src_ptr_array[j]= 2*j + a;
            dsc.dst_ptr_array[j]= 2*j + b[proc];
        }
        {
            armci_giov_t * const iov = &dsc;
            {
                void ** const orig_bufs = iov[0].src_ptr_array;
                const int count = iov[0].ptr_array_len;
                const int size = iov[0].bytes;
                void ** const src_buf = malloc((count+1)*sizeof(void*));
                char *contig;
                MPI_Alloc_mem((MPI_Aint)count*size, MPI_INFO_NULL, &contig);
                src_buf[count] = contig;

                for (int i = 0; i < count; i++) {
                    src_buf[i] = contig + (MPI_Aint)i*size;
                    {
                        const int nelem = size / (int)sizeof(double);
                        double * const s_in = (double*) orig_bufs[i];
                        double * const s_out = (double*) src_buf[i];
                        const double s = alpha;
                        for (int k = 0; k < nelem; k++) s_out[k] = s_in[k] * s;
                    }
                }

                {
                    const int elem_count = iov[0].bytes/(int)sizeof(double);

                    void ** const buf_rem = iov[0].dst_ptr_array;
                    void ** const buf_loc = src_buf;

                    MPI_Datatype  type_loc, type_rem;
                    int           disp_loc[count];
                    int           disp_rem[count];
                    MPI_Aint      loc_addr[count];
                    MPI_Aint      base_loc;
                    MPI_Aint      base_rem;

                    MPI_Get_address(mreg->slices[proc], &base_rem);

                    void *base_loc_ptr = buf_loc[0];
                    MPI_Get_address(buf_loc[0], &base_loc);
                    for (int i = 0; i < count; i++) {
                        MPI_Get_address(buf_loc[i], &loc_addr[i]);
                        if (loc_addr[i] < base_loc) { base_loc = loc_addr[i]; base_loc_ptr = buf_loc[i]; }
                    }

                    for (int i = 0; i < count; i++) {
                        MPI_Aint target_rem;
                        MPI_Get_address(buf_rem[i], &target_rem);
                        disp_rem[i]  = (int)((target_rem - base_rem)/(MPI_Aint)sizeof(double));

                        disp_loc[i]  = (int)((loc_addr[i] - base_loc)/(MPI_Aint)sizeof(double));
                    }

                    for (int start = 0; start < count; start++) {
                        MPI_Type_create_indexed_block(1, elem_count, &disp_loc[start], MPI_DOUBLE, &type_loc);
                        MPI_Type_create_indexed_block(1, elem_count, &disp_rem[start], MPI_DOUBLE, &type_rem);
                        MPI_Type_commit(&type_loc);
                        MPI_Type_commit(&type_rem);

                        MPI_Accumulate(base_loc_ptr, 1, type_loc, proc, 0, 1, type_rem, MPI_SUM, window);

                        MPI_Type_free(&type_loc);
                        MPI_Type_free(&type_rem);
                    }
                    MPI_Win_flush_local(proc, window);
                }

                MPI_Free_mem(src_buf[count]);
                free(src_buf);
            }
        }
        for (int j=0; j<250; j++){
            dsc.src_ptr_array[j]= 2*j+1 + a;
            dsc.dst_ptr_array[j]= 2*j+1 + b[proc];
        }
        {
            armci_giov_t * const iov = &dsc;
            {
                void ** const orig_bufs = iov[0].src_ptr_array;
                const int count = iov[0].ptr_array_len;
                const int size = iov[0].bytes;
                void ** const src_buf = malloc((count+1)*sizeof(void*));
                char *contig;
                MPI_Alloc_mem((MPI_Aint)count*size, MPI_INFO_NULL, &contig);
                src_buf[count] = contig;

                for (int i = 0; i < count; i++) {
                    src_buf[i] = contig + (MPI_Aint)i*size;
                    {
                        const int nelem = size / (int)sizeof(double);
                        double * const s_in = (double*) orig_bufs[i];
                        double * const s_out = (double*) src_buf[i];
                        const double s = alpha;
                        for (int k = 0; k < nelem; k++) s_out[k] = s_in[k] * s;
                    }
                }

                {
                    const int elem_count = iov[0].bytes/(int)sizeof(double);

                    void ** const buf_rem = iov[0].dst_ptr_array;
                    void ** const buf_loc = src_buf;

                    MPI_Datatype  type_loc, type_rem;
                    int           disp_loc[count];
                    int           disp_rem[count];
                    MPI_Aint      loc_addr[count];
                    MPI_Aint      base_loc;
                    MPI_Aint      base_rem;

                    MPI_Get_address(mreg->slices[proc], &base_rem);

                    void *base_loc_ptr = buf_loc[0];
                    MPI_Get_address(buf_loc[0], &base_loc);
                    for (int i = 0; i < count; i++) {
                        MPI_Get_address(buf_loc[i], &loc_addr[i]);
                        if (loc_addr[i] < base_loc) { base_loc = loc_addr[i]; base_loc_ptr = buf_loc[i]; }
                    }

                    for (int i = 0; i < count; i++) {
                        MPI_Aint target_rem;
                        MPI_Get_address(buf_rem[i], &target_rem);
                        disp_rem[i]  = (int)((target_rem - base_rem)/(MPI_Aint)sizeof(double));

                        disp_loc[i]  = (int)((loc_addr[i] - base_loc)/(MPI_Aint)sizeof(double));
                    }

                    for (int start = 0; start < count; start++) {
                        MPI_Type_create_indexed_block(1, elem_count, &disp_loc[start], MPI_DOUBLE, &type_loc);
                        MPI_Type_create_indexed_block(1, elem_count, &disp_rem[start], MPI_DOUBLE, &type_rem);
                        MPI_Type_commit(&type_loc);
                        MPI_Type_commit(&type_rem);

                        MPI_Accumulate(base_loc_ptr, 1, type_loc, proc, 0, 1, type_rem, MPI_SUM, window);

                        MPI_Type_free(&type_loc);
                        MPI_Type_free(&type_rem);
                    }
                    MPI_Win_flush_local(proc, window);
                }

                MPI_Free_mem(src_buf[count]);
                free(src_buf);
            }
        }
    }

    MPI_Win_flush_all(window);
    MPI_Barrier(MPI_COMM_WORLD);

    {
        void * const src = b[proc];
        void * const dst = c;
        const int size = bytes;
        const int target = proc;
        const MPI_Aint disp = (MPI_Aint)((uint8_t*)src -
                                        (uint8_t*)mreg->slices[target]);
        MPI_Get_accumulate(NULL, 0, MPI_BYTE, dst, size, MPI_BYTE, target, disp, size, MPI_BYTE, MPI_NO_OP, window);
        MPI_Win_flush(target, window);
    }

    const double scale = alpha*100*world_np*world_np;
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
