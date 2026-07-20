#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>

#include <mpi.h>

#define MAX(A,B) (((A) > (B)) ? A : B)
#define ABS(a) (((a) <0) ? -(a) : (a))

typedef long armci_size_t;

typedef struct {
    void **src_ptr_array;
    void **dst_ptr_array;
    int    bytes;
    int    ptr_array_len;
} armci_giov_t;

typedef armci_size_t gmr_size_t;

typedef struct gmr_s {
    void **slices;
} gmr_t;

gmr_t *mreg = NULL;
MPI_Win window = MPI_WIN_NULL;

#define ELEMS 500
#define MAXPROC 128
#define TIMES 100
int world_me, world_np;
void* work[MAXPROC];

static void compare_patches(double eps, double *patch1, int lo1[], int hi1[],
                            double *patch2, int lo2[], int hi2[])

{
    int elems = hi1[0]-lo1[0]+1;
    int subscr1 = lo1[0], subscr2 = lo2[0];
    double diff,max;
    int offset1 = 0, offset2 = 0;

    for (int j=0; j< elems; j++){
        int idx1 = subscr1;
        int idx2 = subscr2;

        if(j==0){
            offset1 =idx1;
            offset2 =idx2;
        }
        idx1 -= offset1;
        idx2 -= offset2;

        diff = patch1[idx1] - patch2[idx2];
        max  = MAX(ABS(patch1[idx1]),ABS(patch2[idx2]));
        if(max == 0. || max <eps) max = 1.;

        if(eps < ABS(diff)/max){
            char msg[48];
            sprintf(msg,"(proc=%d):%f",world_me,patch1[idx1]);
            printf("ERROR: a [%d] %s", subscr1, msg);
            sprintf(msg,"%f\n",patch2[idx2]);
            printf(" b [%d] %s", subscr2, msg);
            printf("\nA = %f B = %f\n", patch1[idx1], patch2[idx2]);
            fflush(stdout);
            sleep(1);
            fprintf(stderr, "[%d] ARMCI Error: Bailing out\n", world_me);
            fflush(NULL);
            MPI_Abort(MPI_COMM_WORLD, 0);
        }

        if (subscr1 < hi1[0]) subscr1++;
        else subscr1 = lo1[0];
        if (subscr2 < hi2[0]) subscr2++;
        else subscr2 = lo2[0];
    }

}

static void scale_patch(double alpha, double *patch1, int lo1[], int hi1[])
{
    int elems = hi1[0]-lo1[0]+1;
    int subscr1 = lo1[0];
    int offset1 = 0;

    for (int j=0; j< elems; j++){
        int idx1 = subscr1;

        if(j==0){
            offset1 =idx1;
        }
        idx1 -= offset1;

        patch1[idx1] *= alpha;
        if (subscr1 < hi1[0]) subscr1++;
        else subscr1 = lo1[0];
    }
}

static void GetPermutedProcList(int *ProcList)
{
    int iswap, temp;

    if(world_np > MAXPROC) {
        fprintf(stderr, "[%d] ARMCI Error: permute_proc: nproc to big \n", world_me);
        fflush(NULL);
        MPI_Abort(MPI_COMM_WORLD, world_np);
    }

    for (int i=0; i< world_np; i++) ProcList[i]=i;
    if(world_np ==1) return;

    (void)srand((unsigned)world_me);

    for (int i=0; i< world_np; i++){
        iswap = (int)(rand() % world_np);
        temp = ProcList[iswap];
        ProcList[iswap] = ProcList[i];
        ProcList[i] = temp;
    }
}

static void test_vector_acc(void)
{
    int elems,bytes;
    int proc, one=1;
    double *b[MAXPROC];
    double *a;
    void *c;
    double alpha=0.1, scale;
    int *proclist = (int*)work;
    armci_giov_t dsc = {
        .src_ptr_array = (void *[ELEMS/2]){0},
        .dst_ptr_array = (void *[ELEMS/2]){0},
        .bytes = sizeof(double),
        .ptr_array_len = ELEMS/2
    };

    elems = ELEMS;
    bytes = sizeof(double)*elems;

    {
        void **alloc_slices;

        mreg = malloc(sizeof(gmr_t));
        mreg->slices = malloc(sizeof(void *)*world_np);
        alloc_slices = malloc(sizeof(void *)*world_np);

        MPI_Info win_info = MPI_INFO_NULL;
        MPI_Info_create(&win_info);
        MPI_Info_set(win_info, "alloc_shm", "true");

        MPI_Win_allocate((MPI_Aint)bytes, 1, win_info, MPI_COMM_WORLD,
                         &alloc_slices[world_me], &window);
        MPI_Info_free(&win_info);

        MPI_Allgather(&alloc_slices[world_me], sizeof(void *), MPI_BYTE,
                      alloc_slices, sizeof(void *), MPI_BYTE, MPI_COMM_WORLD);

        for (int gi = 0; gi < world_np; gi++) b[gi] = alloc_slices[gi];

        for (int gi = 0; gi < world_np; gi++) {
            mreg->slices[gi] = alloc_slices[gi];
        }
        free(alloc_slices);

        MPI_Win_lock_all(MPI_MODE_NOCHECK, window);
    }
    a = malloc(bytes);
    c = malloc(bytes);

    for (int i = 0; i < elems; i++) {
        a[i] = i % elems;
    }

    if(world_me==0){
        printf("--------array[%d",elems);
        printf("]--------\n");
        fflush(stdout);
    }

    GetPermutedProcList(proclist);

    for (int i=0;i<elems;i++) b[world_me][i]=0.;

    sleep(1);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0;i<TIMES*world_np;i++){

        proc=0;

        for (int j=0; j<elems/2; j++){
            dsc.src_ptr_array[j]= 2*j + a;
            dsc.dst_ptr_array[j]= 2*j + b[proc];
        }
        {
            armci_giov_t *iov = &dsc;
            int iov_len = 1;
            for (int v = 0; v < iov_len; v++) {
                void **src_buf;

                if (iov[v].ptr_array_len == 0) continue;
                if (iov[v].bytes == 0) continue;

                {
                    void **orig_bufs = iov[v].src_ptr_array;
                    int count = iov[v].ptr_array_len;
                    int size = iov[v].bytes;
                    void **new_bufs;
                    int scaled;

                    new_bufs = malloc((count+1)*sizeof(void*));
                    new_bufs[count] = NULL;

                    scaled = (fabs(alpha - 1.0) < DBL_EPSILON) ? 0 : 1;

                    if (scaled) {
                        char *contig;
                        MPI_Alloc_mem((MPI_Aint)count*size, MPI_INFO_NULL, &contig);
                        new_bufs[count] = contig;

                        for (int i = 0; i < count; i++) {
                            new_bufs[i] = contig + (MPI_Aint)i*size;
                            {
                                int nelem = size / (int)sizeof(double);
                                double *s_in = (double*) orig_bufs[i], *s_out = (double*) new_bufs[i];
                                const double s = alpha;
                                for (int k = 0; k < nelem; k++) s_out[k] = s_in[k] * s;
                            }
                        }
                    } else {
                        for (int i = 0; i < count; i++) {
                            new_bufs[i] = orig_bufs[i];
                        }
                    }

                    src_buf = new_bufs;
                }

                {
                    int elem_count = iov[v].bytes/(int)sizeof(double);

                    void **buf_rem = iov[v].dst_ptr_array;
                    void **buf_loc = src_buf;
                    int count = iov[v].ptr_array_len;

                    MPI_Datatype  type_loc, type_rem;
                    int           disp_loc[count];
                    int           disp_rem[count];
                    MPI_Aint      loc_addr[count];
                    MPI_Aint      base_loc;
                    void         *base_loc_ptr;
                    MPI_Aint      base_rem;

                    MPI_Get_address(mreg->slices[proc], &base_rem);

                    base_loc_ptr = buf_loc[0];
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

                    int chunk = 1;

                    for (int start = 0; start < count; start += chunk) {
                        int n = (count - start < chunk) ? (count - start) : chunk;

                        MPI_Type_create_indexed_block(n, elem_count, &disp_loc[start], MPI_DOUBLE, &type_loc);
                        MPI_Type_create_indexed_block(n, elem_count, &disp_rem[start], MPI_DOUBLE, &type_rem);
                        MPI_Type_commit(&type_loc);
                        MPI_Type_commit(&type_rem);


                        MPI_Accumulate(base_loc_ptr, 1, type_loc, proc, 0, 1, type_rem, MPI_SUM, window);

                        MPI_Type_free(&type_loc);
                        MPI_Type_free(&type_rem);
                    }


                    MPI_Win_flush_local(proc, window);
                }

                {
                    void **orig_bufs = iov[v].src_ptr_array;
                    void **new_bufs  = src_buf;
                    int count = iov[v].ptr_array_len;

                    if (new_bufs[count] != NULL) {
                        MPI_Free_mem(new_bufs[count]);
                    } else {
                        for (int i = 0; i < count; i++) {
                            if (orig_bufs[i] != new_bufs[i]) {
                                MPI_Free_mem(new_bufs[i]);
                            }
                        }
                    }

                    free(new_bufs);
                }
            }

        }
        for (int j=0; j< elems/2; j++){
            dsc.src_ptr_array[j]= 2*j+1 + a;
            dsc.dst_ptr_array[j]= 2*j+1 + b[proc];
        }
        {
            armci_giov_t *iov = &dsc;
            int iov_len = 1;
            for (int v = 0; v < iov_len; v++) {
                void **src_buf;

                if (iov[v].ptr_array_len == 0) continue;
                if (iov[v].bytes == 0) continue;

                {
                    void **orig_bufs = iov[v].src_ptr_array;
                    int count = iov[v].ptr_array_len;
                    int size = iov[v].bytes;
                    void **new_bufs;
                    int scaled;

                    new_bufs = malloc((count+1)*sizeof(void*));
                    new_bufs[count] = NULL;

                    scaled = (fabs(alpha - 1.0) < DBL_EPSILON) ? 0 : 1;

                    if (scaled) {
                        char *contig;
                        MPI_Alloc_mem((MPI_Aint)count*size, MPI_INFO_NULL, &contig);
                        new_bufs[count] = contig;

                        for (int i = 0; i < count; i++) {
                            new_bufs[i] = contig + (MPI_Aint)i*size;
                            {
                                int nelem = size / (int)sizeof(double);
                                double *s_in = (double*) orig_bufs[i], *s_out = (double*) new_bufs[i];
                                const double s = alpha;
                                for (int k = 0; k < nelem; k++) s_out[k] = s_in[k] * s;
                            }
                        }
                    } else {
                        for (int i = 0; i < count; i++) {
                            new_bufs[i] = orig_bufs[i];
                        }
                    }

                    src_buf = new_bufs;
                }

                {
                    int elem_count = iov[v].bytes/(int)sizeof(double);

                    void **buf_rem = iov[v].dst_ptr_array;
                    void **buf_loc = src_buf;
                    int count = iov[v].ptr_array_len;

                    MPI_Datatype  type_loc, type_rem;
                    int           disp_loc[count];
                    int           disp_rem[count];
                    MPI_Aint      loc_addr[count];
                    MPI_Aint      base_loc;
                    void         *base_loc_ptr;
                    MPI_Aint      base_rem;

                    MPI_Get_address(mreg->slices[proc], &base_rem);

                    base_loc_ptr = buf_loc[0];
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

                    int chunk = 1;

                    for (int start = 0; start < count; start += chunk) {
                        int n = (count - start < chunk) ? (count - start) : chunk;

                        MPI_Type_create_indexed_block(n, elem_count, &disp_loc[start], MPI_DOUBLE, &type_loc);
                        MPI_Type_create_indexed_block(n, elem_count, &disp_rem[start], MPI_DOUBLE, &type_rem);
                        MPI_Type_commit(&type_loc);
                        MPI_Type_commit(&type_rem);


                        MPI_Accumulate(base_loc_ptr, 1, type_loc, proc, 0, 1, type_rem, MPI_SUM, window);

                        MPI_Type_free(&type_loc);
                        MPI_Type_free(&type_rem);
                    }


                    MPI_Win_flush_local(proc, window);
                }

                {
                    void **orig_bufs = iov[v].src_ptr_array;
                    void **new_bufs  = src_buf;
                    int count = iov[v].ptr_array_len;

                    if (new_bufs[count] != NULL) {
                        MPI_Free_mem(new_bufs[count]);
                    } else {
                        for (int i = 0; i < count; i++) {
                            if (orig_bufs[i] != new_bufs[i]) {
                                MPI_Free_mem(new_bufs[i]);
                            }
                        }
                    }

                    free(new_bufs);
                }
            }

        }

    }

    MPI_Win_flush_all(window);
    MPI_Barrier(MPI_COMM_WORLD);

    {
        void *src = b[proc];
        void *dst = c;
        int size = bytes;
        int target = proc;

        if (target == world_me) {
            memmove(dst, src, size);
        }
        else {
            {
                gmr_size_t disp;
                if (src == MPI_BOTTOM) disp = 0;
                else disp = (gmr_size_t)((uint8_t*)src - (uint8_t*)mreg->slices[target]);
                MPI_Get_accumulate(NULL, 0, MPI_BYTE, dst, size, MPI_BYTE, target,
                                   (MPI_Aint) disp, size, MPI_BYTE, MPI_NO_OP, window);
            }
            {
                MPI_Win_flush(target, window);
            }
        }
    }

    scale = alpha*TIMES*world_np*world_np;
    scale_patch(scale, a, &one, &elems);

    compare_patches(.0001, a, &one, &elems, c, &one, &elems);
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
    mreg = NULL;
    free(a);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_me);
    MPI_Comm_size(MPI_COMM_WORLD, &world_np);
    if (world_me == 0) printf("standalone test_vector_acc: %d procs\n", world_np);
    test_vector_acc();
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_me == 0) printf("DONE OK\n");
    MPI_Finalize();
    return 0;
}
