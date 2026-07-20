#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <unistd.h>

#include <mpi.h>

#define MAX(A,B) (((A) > (B)) ? A : B)
#define MIN(A,B) (((A) < (B)) ? (A) : (B))
#define ABS(a) (((a) <0) ? -(a) : (a))

typedef long armci_size_t;
#define ARMCII_MPI_SIZE_T MPI_LONG

void  ARMCI_Error(const char *msg, int code);

typedef struct {
  void **src_ptr_array;
  void **dst_ptr_array;
  int    bytes;
  int    ptr_array_len;
} armci_giov_t;

typedef struct {
  MPI_Comm  comm;
  MPI_Comm  noncoll_pgroup_comm;
  int      *grp_to_abs;
  int      *abs_to_grp;
  int       rank;
  int       size;
} ARMCI_Group;

extern ARMCI_Group ARMCI_GROUP_WORLD;
extern ARMCI_Group ARMCI_GROUP_DEFAULT;





#define ARMCII_Error(...) ARMCII_Error_impl(__FILE__,__LINE__,__func__,__VA_ARGS__)
void    ARMCII_Error_impl(const char *file, const int line, const char *func, const char *msg, ...);
void    ARMCII_Warning(const char *fmt, ...);

typedef armci_size_t gmr_size_t;
#define GMR_MPI_SIZE_T ARMCII_MPI_SIZE_T

typedef struct {
  void       *base;
  gmr_size_t  size;
} gmr_slice_t;

typedef struct gmr_s {
  ARMCI_Group             group;

  struct gmr_s           *prev;
  struct gmr_s           *next;
  gmr_slice_t            *slices;
  int                     nslices;
  bool                    unified;
} gmr_t;

extern gmr_t *gmr_list;




void ARMCII_Error_impl(const char *file, const int line, const char *func, const char *msg, ...) {
  va_list ap;
  int  disp;
  char string[500];

  disp  = 0;
  va_start(ap, msg);
  disp += vsnprintf(string, 500, msg, ap);
  va_end(ap);

  fprintf(stderr, "[%d] ARMCI Internal error in %s (%s:%d)\n[%d] Message: %s\n", ARMCI_GROUP_WORLD.rank,
      func, file, line, ARMCI_GROUP_WORLD.rank, string);
  MPI_Abort(ARMCI_GROUP_WORLD.comm, 100);
}

void ARMCII_Warning(const char *fmt, ...) {
  va_list etc;
  int  disp;
  char string[500];

  disp  = 0;
  disp += snprintf(string, 500, "[%d] ARMCI Warning: ", ARMCI_GROUP_WORLD.rank);
  va_start(etc, fmt);
  disp += vsnprintf(string+disp, 500-disp, fmt, etc);
  va_end(etc);

  fprintf(stderr, "%s", string);
  fflush(NULL);
}

void ARMCI_Error(const char *msg, int code) {
  fprintf(stderr, "[%d] ARMCI Error: %s\n", ARMCI_GROUP_WORLD.rank, msg);
  fflush(NULL);
  MPI_Abort(ARMCI_GROUP_WORLD.comm, code);
}

ARMCI_Group ARMCI_GROUP_WORLD   = {0};
ARMCI_Group ARMCI_GROUP_DEFAULT = {0};

gmr_t *gmr_list = NULL;
MPI_Win window = MPI_WIN_NULL;   /* single global window */

/* DO NOT INLINE ANYTHING BELOW HERE */

#define ELEMS 500
#define MAXPROC 128
#define TIMES 100
#define BASE 100.
#define MAXDIMS 7
int me, nproc;
void* work[MAXPROC];
static void print_subscript(char *pre,int ndim, int subscript[], char* post)
{
    int i;
    printf("%s [",pre);
    for(i=0;i<ndim;i++){ printf("%d",subscript[i]); if(i==ndim-1)printf("] %s",post); else printf(","); }
}

void init(double *a, int ndim, int elems, int dims[])
{
  int idx[MAXDIMS];
  int i,dim;

 	for(i=0; i<elems; i++){
		int Index = i;
		double field, val;

		for(dim = 0; dim < ndim; dim++){
			idx[dim] = Index%dims[dim];
			Index /= dims[dim];
		}

        field=1.; val=0.;
		for(dim=0; dim< ndim;dim++){
			val += field*idx[dim];
			field *= BASE;
		}
		a[i] = val;

	}
}

int Index(int ndim, int subscript[], int dims[])
{
	int idx = 0, i, factor=1;
	for(i=0;i<ndim;i++){
		idx += subscript[i]*factor;
		factor *= dims[i];
	}
	return idx;
}

void update_subscript(int ndim, int subscript[], int lo[], int hi[], int dims[])
{
	int i;
	for(i=0;i<ndim;i++){
		if(subscript[i] < hi[i]) { subscript[i]++; return; }
		subscript[i] = lo[i];
	}
}

void compare_patches(double eps, int ndim, double *patch1, int lo1[], int hi1[],
                     int dims1[],double *patch2, int lo2[], int hi2[],
                     int dims2[])

{
	int i,j, elems=1;
	int subscr1[MAXDIMS], subscr2[MAXDIMS];
        double diff,max;
    int offset1 = 0, offset2 = 0;

	for(i=0;i<ndim;i++){
		int diff = hi1[i]-lo1[i];
		elems *= diff+1;
		subscr1[i]= lo1[i];
		subscr2[i]=lo2[i];
	}

	for(j=0; j< elems; j++){
		int idx1, idx2;

		idx1 = Index(ndim, subscr1, dims1);
		idx2 = Index(ndim, subscr2, dims2);

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
			sprintf(msg,"(proc=%d):%f",me,patch1[idx1]);
			print_subscript("ERROR: a",ndim,subscr1,msg);
			sprintf(msg,"%f\n",patch2[idx2]);
			print_subscript(" b",ndim,subscr2,msg);
                        printf("\nA = %f B = %f\n", patch1[idx1], patch2[idx2]);
                        fflush(stdout);
                        sleep(1);
                        ARMCI_Error("Bailing out",0);
		}

		{
		   update_subscript(ndim, subscr1, lo1,hi1, dims1);
		   update_subscript(ndim, subscr2, lo2,hi2, dims2);
		}
	}

}

void scale_patch(double alpha, int ndim, double *patch1, int lo1[], int hi1[], int dims1[])
{
	int i,j, elems=1;
	int subscr1[MAXDIMS];
	int offset1 = 0;

	for(i=0;i<ndim;i++){
		int diff = hi1[i]-lo1[i];
		elems *= diff+1;
		subscr1[i]= lo1[i];
	}

	for(j=0; j< elems; j++){
		int idx1;

		idx1 = Index(ndim, subscr1, dims1);

		if(j==0){
			offset1 =idx1;
		}
		idx1 -= offset1;

		patch1[idx1] *= alpha;
		update_subscript(ndim, subscr1, lo1,hi1, dims1);
	}
}

void create_array(void *a[], int elem_size, int ndim, int dims[])
{
     int bytes=elem_size, i, rc;

     for(i=0;i<ndim;i++)bytes*=dims[i];
     /* inlined PARMCI_Malloc -> ARMCI_Malloc_group -> gmr_create */
     { /* inlined gmr_create(bytes, a, &ARMCI_GROUP_WORLD) */
       gmr_size_t local_size = bytes;
       void **base_ptrs = a;
       ARMCI_Group *group = &ARMCI_GROUP_WORLD;
       int           gi, alloc_me, alloc_nproc, world_me, world_nproc;
       MPI_Group     world_group, alloc_group;
       gmr_t        *mreg;
       gmr_slice_t  *alloc_slices, gmr_slice;


       MPI_Comm_rank(group->comm, &alloc_me);
       MPI_Comm_size(group->comm, &alloc_nproc);

       gmr_size_t max_local_size;
       MPI_Allreduce(&local_size, &max_local_size, 1, GMR_MPI_SIZE_T, MPI_MAX, group->comm);

       if (max_local_size == 0) {
         for (gi = 0; gi < alloc_nproc; gi++) base_ptrs[gi] = NULL;
       } else {
         MPI_Comm_rank(ARMCI_GROUP_WORLD.comm, &world_me);
         MPI_Comm_size(ARMCI_GROUP_WORLD.comm, &world_nproc);

         mreg = malloc(sizeof(gmr_t));
         mreg->slices = malloc(sizeof(gmr_slice_t)*world_nproc);
         alloc_slices = malloc(sizeof(gmr_slice_t)*alloc_nproc);

         mreg->group   = *group;
         mreg->nslices = world_nproc;
         mreg->prev    = NULL;
         mreg->next    = NULL;
         mreg->unified = false;

         alloc_slices[alloc_me].size = local_size;

         MPI_Info win_info = MPI_INFO_NULL;
         MPI_Info_create(&win_info);
         MPI_Info_set(win_info, "alloc_shm", "true");   /* use_alloc_shm */

         /* use_win_allocate == 1 */
         MPI_Win_allocate( (MPI_Aint) local_size, 1, win_info, group->comm, &(alloc_slices[alloc_me].base), &window);
         if (local_size == 0) alloc_slices[alloc_me].base = NULL;

         MPI_Info_free(&win_info);

         gmr_slice = alloc_slices[alloc_me];
         MPI_Allgather(&gmr_slice, sizeof(gmr_slice_t), MPI_BYTE,
                       alloc_slices, sizeof(gmr_slice_t), MPI_BYTE, group->comm);

         for (gi = 0; gi < alloc_nproc; gi++) base_ptrs[gi] = alloc_slices[gi].base;

         memset(mreg->slices, 0, sizeof(gmr_slice_t)*world_nproc);

         MPI_Comm_group(ARMCI_GROUP_WORLD.comm, &world_group);
         MPI_Comm_group(group->comm, &alloc_group);
         for (gi = 0; gi < alloc_nproc; gi++) {
           int world_rank;
           MPI_Group_translate_ranks(alloc_group, 1, &gi, world_group, &world_rank);
           mreg->slices[world_rank] = alloc_slices[gi];
         }
         free(alloc_slices);
         MPI_Group_free(&world_group);
         MPI_Group_free(&alloc_group);

         MPI_Win_lock_all(MPI_MODE_NOCHECK, window);   /* rma_nocheck */

         {
           int unified;
           { void *attr_ptr; int attr_flag;
             MPI_Win_get_attr(window, MPI_WIN_MODEL, &attr_ptr, &attr_flag);
             if (attr_flag) { int *attr_val = (int*)attr_ptr;
               if      ((*attr_val)==MPI_WIN_UNIFIED) unified = 1;
               else if ((*attr_val)==MPI_WIN_UNIFIED) unified = 0;
               else                                   unified = -1;
             } else unified = -1;
           }
           mreg->unified = (unified == 1);                    /* verbose=0: banner prints dropped */
           if (!(mreg->unified)) {                            /* shr_buf_method == NOGUARD */
             if (world_me==0) printf("Please re-run with ARMCI_SHR_BUF_METHOD=COPY\n");
           }
         }

         if (gmr_list == NULL) {
           gmr_list = mreg;
         } else {
           gmr_t *parent = gmr_list;
           while (parent->next != NULL) parent = parent->next;
           parent->next = mreg;
           mreg->prev   = parent;
         }
       }
     }
     rc = 0;

}

void destroy_array(void *ptr[])
{
    int rc;
    MPI_Barrier(MPI_COMM_WORLD);
    { /* inlined PARMCI_Free -> ARMCI_Free_group */
      gmr_t *mreg;
      void *fptr = ptr[me];
      if (fptr != NULL) {
        /* inlined gmr_lookup(fptr, ARMCI_GROUP_WORLD.rank) */
        void *lk_ptr = (fptr); int lk_proc = (ARMCI_GROUP_WORLD.rank);
        gmr_t *m = gmr_list;
        while (m != NULL) {
          if (lk_proc < m->nslices) {
            const uint8_t   *base = m->slices[lk_proc].base;
            const gmr_size_t sz   = m->slices[lk_proc].size;
            if ((uint8_t*)lk_ptr >= base && (uint8_t*)lk_ptr < base + sz) break;
          }
          m = m->next;
        }
        mreg = m;
      } else {
        mreg = NULL;
      }
      { /* inlined gmr_destroy(mreg, &ARMCI_GROUP_WORLD) */
        ARMCI_Group *group = &ARMCI_GROUP_WORLD;
        int search_proc_in, search_proc_out, search_proc_out_grp;
        void *search_base = NULL;
        int alloc_me, alloc_nproc, world_me, world_nproc;
        MPI_Comm_rank(group->comm, &alloc_me);
        MPI_Comm_size(group->comm, &alloc_nproc);
        MPI_Comm_rank(ARMCI_GROUP_WORLD.comm, &world_me);
        MPI_Comm_size(ARMCI_GROUP_WORLD.comm, &world_nproc);
        if (mreg == NULL) search_proc_in = -1;
        else { search_proc_in = world_me; search_base = mreg->slices[world_me].base; }
        MPI_Allreduce(&search_proc_in, &search_proc_out, 1, MPI_INT, MPI_MAX, group->comm);
        if (search_proc_out >= 0) {
          search_proc_out_grp = search_proc_out;
          MPI_Bcast(&search_base, sizeof(void*), MPI_BYTE, search_proc_out_grp, group->comm);
          if (mreg == NULL) {
            void *lk_ptr = (search_base); int lk_proc = (search_proc_out);
            gmr_t *m = gmr_list;
            while (m != NULL) {
              if (lk_proc < m->nslices) {
                const uint8_t   *base = m->slices[lk_proc].base;
                const gmr_size_t sz   = m->slices[lk_proc].size;
                if ((uint8_t*)lk_ptr >= base && (uint8_t*)lk_ptr < base + sz) break;
              }
              m = m->next;
            }
            mreg = m;
          }
          if (mreg->prev == NULL) {
            gmr_list = mreg->next;
            if (mreg->next != NULL) mreg->next->prev = NULL;
          } else {
            mreg->prev->next = mreg->next;
            if (mreg->next != NULL) mreg->next->prev = mreg->prev;
          }
          MPI_Win_unlock_all(window);
          MPI_Win_free(&window);
          free(mreg->slices);
          free(mreg);
        }
      }
      rc = 0;
    }
}

void GetPermutedProcList(int* ProcList)
{
    int i, iswap, temp;

    if(nproc > MAXPROC) ARMCI_Error("permute_proc: nproc to big ", nproc);

    for(i=0; i< nproc; i++) ProcList[i]=i;
    if(nproc ==1) return;

    (void)srand((unsigned)me);

    for(i=0; i< nproc; i++){
      iswap = (int)(rand() % nproc);
      temp = ProcList[iswap];
      ProcList[iswap] = ProcList[i];
      ProcList[i] = temp;
    }
}

void test_vector_acc()
{
	int dim,elems,bytes;
	int i, j, proc, rc, one=1;
        void *b[MAXPROC];
        void *psrc[ELEMS/2], *pdst[ELEMS/2];
        void *a, *c;
        double alpha=0.1, scale;
        int *proclist = (int*)work;
        armci_giov_t dsc;

        elems = ELEMS;
        dim =1;
        bytes = sizeof(double)*elems;

        create_array((void**)b, sizeof(double),dim,&elems);
        a = malloc(bytes);
        c = malloc(bytes);

	init(a, dim, elems, &elems);

	if(me==0){
            printf("--------array[%d",elems);
	    printf("]--------\n");
            fflush(stdout);
        }

        GetPermutedProcList(proclist);

        for(i=0;i<elems;i++)((double*)b[me])[i]=0.;

        sleep(1);

        dsc.bytes = sizeof(double);
        dsc.src_ptr_array = psrc;
        dsc.dst_ptr_array = pdst;
        dsc.ptr_array_len = elems/2;

        MPI_Barrier(MPI_COMM_WORLD);
        for(i=0;i<TIMES*nproc;i++){

            proc=0;

            for(j=0; j<elems/2; j++){
                psrc[j]= 2*j + (double*)a;
                pdst[j]= 2*j + (double*)b[proc];
            }
            { /* inlined PARMCI_AccV(ARMCI_ACC_DBL, &alpha, &dsc, 1, proc) */
              void *acc_scale = &alpha;
              armci_giov_t *iov = &dsc;
              int iov_len = 1;
              for (int v = 0; v < iov_len; v++) {
                void **src_buf;
            
                if (iov[v].ptr_array_len == 0) continue;
                if (iov[v].bytes == 0) continue;
            
                { /* inlined ARMCII_Buf_prepare_acc_vec(iov[v].src_ptr_array, &src_buf, ...) */
                  void **orig_bufs = iov[v].src_ptr_array;
                  int count = iov[v].ptr_array_len;
                  int size = iov[v].bytes;
                  void **new_bufs;
                  int i, scaled;
            
                  new_bufs = malloc((count+1)*sizeof(void*));
                  new_bufs[count] = NULL;
            
                  scaled = (fabs(*((double*)acc_scale) - 1.0) < DBL_EPSILON) ? 0 : 1;   /* inlined ARMCII_Buf_acc_is_scaled (ARMCI_ACC_DBL) */
            
                  if (scaled) {
                    char *contig;
                    MPI_Alloc_mem((MPI_Aint)count*size, MPI_INFO_NULL, &contig);
                    new_bufs[count] = contig;
            
                    for (i = 0; i < count; i++) {
                      new_bufs[i] = contig + (MPI_Aint)i*size;
                      { /* inlined ARMCII_Buf_acc_scale (ARMCI_ACC_DBL) */
                        int nelem = size / (int)sizeof(double);
                        double *s_in = (double*) orig_bufs[i], *s_out = (double*) new_bufs[i];
                        const double s = *((double*) acc_scale);
                        for (int k = 0; k < nelem; k++) s_out[k] = s_in[k] * s;
                      }
                    }
                  } else {
                    for (i = 0; i < count; i++) {
                      new_bufs[i] = orig_bufs[i];   /* shr_buf=NOGUARD: no guard copy */
                    }
                  }
            
                  src_buf = new_bufs;
                }
            
                { /* inlined ARMCII_Iov_op_dispatch + ARMCII_Iov_op_datatype (op = ACC, blocking = 1) */
                  MPI_Datatype type = MPI_DOUBLE;   /* ARMCI_ACC_DBL */
                  int type_size;
                  MPI_Type_size(type, &type_size);
                  int elem_count = iov[v].bytes/type_size;
            
                  void **buf_rem = iov[v].dst_ptr_array;   /* ACC: remote = dst */
                  void **buf_loc = src_buf;                /* ACC: local  = src */
                  int count = iov[v].ptr_array_len;
            
                  gmr_t *mreg;
                  MPI_Datatype  type_loc, type_rem;
                  int           disp_loc[count];
                  int           disp_rem[count];
                  MPI_Aint      loc_addr[count];
                  MPI_Aint      base_loc;
                  void         *base_loc_ptr;
                  void         *dst_win_base;
                  int           dst_win_size, i;
                  MPI_Aint      base_rem;
            
                  { /* inlined gmr_lookup(buf_rem[0], proc) */
                    void *lk_ptr = (buf_rem[0]); int lk_proc = (proc);
                    gmr_t *m = gmr_list;
                    while (m != NULL) {
                      if (lk_proc < m->nslices) {
                        const uint8_t   *base = m->slices[lk_proc].base;
                        const gmr_size_t sz   = m->slices[lk_proc].size;
                        if ((uint8_t*)lk_ptr >= base && (uint8_t*)lk_ptr < base + sz) break;
                      }
                      m = m->next;
                    }
                    mreg = m;
                  }
            
                  dst_win_base = mreg->slices[proc].base;
                  dst_win_size = mreg->slices[proc].size;
            
                  MPI_Get_address(dst_win_base, &base_rem);
            
                  base_loc_ptr = buf_loc[0];
                  MPI_Get_address(buf_loc[0], &base_loc);
                  for (i = 0; i < count; i++) {
                    MPI_Get_address(buf_loc[i], &loc_addr[i]);
                    if (loc_addr[i] < base_loc) { base_loc = loc_addr[i]; base_loc_ptr = buf_loc[i]; }
                  }
            
                  for (i = 0; i < count; i++) {
                    MPI_Aint target_rem, off_loc;
                    MPI_Get_address(buf_rem[i], &target_rem);
                    off_loc      = (loc_addr[i] - base_loc)/type_size;
                    disp_rem[i]  = (target_rem - base_rem)/type_size;
            
                    disp_loc[i]  = (int)off_loc;
                  }
            
                  int chunk = 1;   /* iov_dtype_chunk; count>=1, no clamp */
            
                  for (int start = 0; start < count; start += chunk) {
                    int n = (count - start < chunk) ? (count - start) : chunk;
            
                    MPI_Type_create_indexed_block(n, elem_count, &disp_loc[start], type, &type_loc);
                    MPI_Type_create_indexed_block(n, elem_count, &disp_rem[start], type, &type_rem);
                    MPI_Type_commit(&type_loc);
                    MPI_Type_commit(&type_rem);
            
                    /* origin = base_loc_ptr + type_loc; target = window base (disp 0) + type_rem,
                     * whose element offsets are relative to the window base. */
                    MPI_Accumulate(base_loc_ptr, 1, type_loc, proc, 0, 1, type_rem, MPI_SUM, window);
            
                    MPI_Type_free(&type_loc);
                    MPI_Type_free(&type_rem);
                  }
            
                  /* blocking=1, flush_local=1 (ACC): inlined gmr_flush(mreg, proc, 1) */
                  MPI_Win_flush_local(proc, window);   /* end_to_end_flush=0 */
                }
            
                { /* inlined ARMCII_Buf_finish_acc_vec(iov[v].src_ptr_array, src_buf, ...) */
                  void **orig_bufs = iov[v].src_ptr_array;
                  void **new_bufs  = src_buf;
                  int count = iov[v].ptr_array_len;
                  int i;
            
                  if (new_bufs[count] != NULL) {
                    MPI_Free_mem(new_bufs[count]);
                  } else {
                    for (i = 0; i < count; i++) {
                      if (orig_bufs[i] != new_bufs[i]) {
                        MPI_Free_mem(new_bufs[i]);
                      }
                    }
                  }
            
                  free(new_bufs);
                }
              }
            
            }
            rc = 0;   /* PARMCI_AccV always returns 0 */

            for(j=0; j< elems/2; j++){
                psrc[j]= 2*j+1 + (double*)a;
                pdst[j]= 2*j+1 + (double*)b[proc];
            }
            { /* inlined PARMCI_AccV(ARMCI_ACC_DBL, &alpha, &dsc, 1, proc) */
              void *acc_scale = &alpha;
              armci_giov_t *iov = &dsc;
              int iov_len = 1;
              for (int v = 0; v < iov_len; v++) {
                void **src_buf;
            
                if (iov[v].ptr_array_len == 0) continue;
                if (iov[v].bytes == 0) continue;
            
                { /* inlined ARMCII_Buf_prepare_acc_vec(iov[v].src_ptr_array, &src_buf, ...) */
                  void **orig_bufs = iov[v].src_ptr_array;
                  int count = iov[v].ptr_array_len;
                  int size = iov[v].bytes;
                  void **new_bufs;
                  int i, scaled;
            
                  new_bufs = malloc((count+1)*sizeof(void*));
                  new_bufs[count] = NULL;
            
                  scaled = (fabs(*((double*)acc_scale) - 1.0) < DBL_EPSILON) ? 0 : 1;   /* inlined ARMCII_Buf_acc_is_scaled (ARMCI_ACC_DBL) */
            
                  if (scaled) {
                    char *contig;
                    MPI_Alloc_mem((MPI_Aint)count*size, MPI_INFO_NULL, &contig);
                    new_bufs[count] = contig;
            
                    for (i = 0; i < count; i++) {
                      new_bufs[i] = contig + (MPI_Aint)i*size;
                      { /* inlined ARMCII_Buf_acc_scale (ARMCI_ACC_DBL) */
                        int nelem = size / (int)sizeof(double);
                        double *s_in = (double*) orig_bufs[i], *s_out = (double*) new_bufs[i];
                        const double s = *((double*) acc_scale);
                        for (int k = 0; k < nelem; k++) s_out[k] = s_in[k] * s;
                      }
                    }
                  } else {
                    for (i = 0; i < count; i++) {
                      new_bufs[i] = orig_bufs[i];   /* shr_buf=NOGUARD: no guard copy */
                    }
                  }
            
                  src_buf = new_bufs;
                }
            
                { /* inlined ARMCII_Iov_op_dispatch + ARMCII_Iov_op_datatype (op = ACC, blocking = 1) */
                  MPI_Datatype type = MPI_DOUBLE;   /* ARMCI_ACC_DBL */
                  int type_size;
                  MPI_Type_size(type, &type_size);
                  int elem_count = iov[v].bytes/type_size;
            
                  void **buf_rem = iov[v].dst_ptr_array;   /* ACC: remote = dst */
                  void **buf_loc = src_buf;                /* ACC: local  = src */
                  int count = iov[v].ptr_array_len;
            
                  gmr_t *mreg;
                  MPI_Datatype  type_loc, type_rem;
                  int           disp_loc[count];
                  int           disp_rem[count];
                  MPI_Aint      loc_addr[count];
                  MPI_Aint      base_loc;
                  void         *base_loc_ptr;
                  void         *dst_win_base;
                  int           dst_win_size, i;
                  MPI_Aint      base_rem;
            
                  { /* inlined gmr_lookup(buf_rem[0], proc) */
                    void *lk_ptr = (buf_rem[0]); int lk_proc = (proc);
                    gmr_t *m = gmr_list;
                    while (m != NULL) {
                      if (lk_proc < m->nslices) {
                        const uint8_t   *base = m->slices[lk_proc].base;
                        const gmr_size_t sz   = m->slices[lk_proc].size;
                        if ((uint8_t*)lk_ptr >= base && (uint8_t*)lk_ptr < base + sz) break;
                      }
                      m = m->next;
                    }
                    mreg = m;
                  }
            
                  dst_win_base = mreg->slices[proc].base;
                  dst_win_size = mreg->slices[proc].size;
            
                  MPI_Get_address(dst_win_base, &base_rem);
            
                  base_loc_ptr = buf_loc[0];
                  MPI_Get_address(buf_loc[0], &base_loc);
                  for (i = 0; i < count; i++) {
                    MPI_Get_address(buf_loc[i], &loc_addr[i]);
                    if (loc_addr[i] < base_loc) { base_loc = loc_addr[i]; base_loc_ptr = buf_loc[i]; }
                  }
            
                  for (i = 0; i < count; i++) {
                    MPI_Aint target_rem, off_loc;
                    MPI_Get_address(buf_rem[i], &target_rem);
                    off_loc      = (loc_addr[i] - base_loc)/type_size;
                    disp_rem[i]  = (target_rem - base_rem)/type_size;
            
                    disp_loc[i]  = (int)off_loc;
                  }
            
                  int chunk = 1;   /* iov_dtype_chunk; count>=1, no clamp */
            
                  for (int start = 0; start < count; start += chunk) {
                    int n = (count - start < chunk) ? (count - start) : chunk;
            
                    MPI_Type_create_indexed_block(n, elem_count, &disp_loc[start], type, &type_loc);
                    MPI_Type_create_indexed_block(n, elem_count, &disp_rem[start], type, &type_rem);
                    MPI_Type_commit(&type_loc);
                    MPI_Type_commit(&type_rem);
            
                    /* origin = base_loc_ptr + type_loc; target = window base (disp 0) + type_rem,
                     * whose element offsets are relative to the window base. */
                    MPI_Accumulate(base_loc_ptr, 1, type_loc, proc, 0, 1, type_rem, MPI_SUM, window);
            
                    MPI_Type_free(&type_loc);
                    MPI_Type_free(&type_rem);
                  }
            
                  /* blocking=1, flush_local=1 (ACC): inlined gmr_flush(mreg, proc, 1) */
                  MPI_Win_flush_local(proc, window);   /* end_to_end_flush=0 */
                }
            
                { /* inlined ARMCII_Buf_finish_acc_vec(iov[v].src_ptr_array, src_buf, ...) */
                  void **orig_bufs = iov[v].src_ptr_array;
                  void **new_bufs  = src_buf;
                  int count = iov[v].ptr_array_len;
                  int i;
            
                  if (new_bufs[count] != NULL) {
                    MPI_Free_mem(new_bufs[count]);
                  } else {
                    for (i = 0; i < count; i++) {
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

        MPI_Win_flush_all(window);   /* inlined PARMCI_AllFence(): single global window */
        MPI_Barrier(MPI_COMM_WORLD);

	{ /* inlined PARMCI_Get((double*)b[proc], c, bytes, proc) (always returns 0) */
	  void *src = (double*)b[proc];
	  void *dst = c;
	  int size = bytes;
	  int target = proc;
	  gmr_t *src_mreg;

	  { /* inlined gmr_lookup(src, target) */
	    void *lk_ptr = (src); int lk_proc = (target);
	    gmr_t *m = gmr_list;
	    while (m != NULL) {
	      if (lk_proc < m->nslices) {
	        const uint8_t   *base = m->slices[lk_proc].base;
	        const gmr_size_t sz   = m->slices[lk_proc].size;
	        if ((uint8_t*)lk_ptr >= base && (uint8_t*)lk_ptr < base + sz) break;
	      }
	      m = m->next;
	    }
	    src_mreg = m;
	  }

	  /* dst_mreg == NULL always (shr_buf=NOGUARD) */
	  if (target == ARMCI_GROUP_WORLD.rank) {
	    memmove(dst, src, size);
	  }
	  else {
	    { /* inlined gmr_get_typed(...) */
	      gmr_t *gt_mreg = src_mreg;
	      int grp_proc = target;
	      gmr_size_t disp;
	      MPI_Aint lb, extent;
	      if (src == MPI_BOTTOM) disp = 0;
	      else disp = (gmr_size_t)((uint8_t*)src - (uint8_t*)gt_mreg->slices[target].base);
	      MPI_Type_get_true_extent(MPI_BYTE, &lb, &extent);
	      MPI_Get_accumulate(NULL, 0, MPI_BYTE, dst, size, MPI_BYTE, grp_proc,
	                           (MPI_Aint) disp, size, MPI_BYTE, MPI_NO_OP, window);   /* rma_atomicity */
	    }
	    { /* inlined gmr_flush(src_mreg, target, 0) -> full flush */
	      MPI_Win_flush(target, window);
	    }
	  }
	}

        scale = alpha*TIMES*nproc*nproc;
        scale_patch(scale, dim, a, &one, &elems, &elems);

        compare_patches(.0001, dim, a, &one, &elems, &elems, c, &one, &elems, &elems);
        MPI_Barrier(MPI_COMM_WORLD);

        if(0==me){
            printf(" OK\n\n");
            fflush(stdout);
        }

        free(c);
        destroy_array((void**)b);
        free(a);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    { /* inlined PARMCI_Init_thread_comm(MPI_THREAD_SINGLE, MPI_COMM_WORLD); init_count folded (called once) */
      MPI_Comm comm = MPI_COMM_WORLD;
      {
        int mpi_is_init, mpi_is_fin;
        MPI_Initialized(&mpi_is_init);
        MPI_Finalized(&mpi_is_fin);
        if (!mpi_is_init || mpi_is_fin) {
          ARMCII_Error("MPI must be initialized before calling ARMCI_Init");
        }
      }

      MPI_Comm_dup(comm, &ARMCI_GROUP_WORLD.comm);
      { /* inlined ARMCII_Group_init_from_comm(&ARMCI_GROUP_WORLD) */
        ARMCI_Group *group = &ARMCI_GROUP_WORLD;
        if (group->comm != MPI_COMM_NULL) {
          MPI_Comm_size(group->comm, &group->size);
          MPI_Comm_rank(group->comm, &group->rank);
        } else {
          group->rank = -1;
          group->size =  0;
        }
        group->noncoll_pgroup_comm = MPI_COMM_NULL;
        group->abs_to_grp = NULL;
        group->grp_to_abs = NULL;
      }
      ARMCI_GROUP_DEFAULT = ARMCI_GROUP_WORLD;
    }
    if (me == 0) printf("standalone test_vector_acc: %d procs\n", nproc);
    test_vector_acc();
    MPI_Barrier(MPI_COMM_WORLD);
    if (me == 0) printf("DONE OK\n");
    { /* inlined PARMCI_Finalize() */
      int nfreed;
      nfreed = 0;   /* inlined gmr_destroy_all() */
      while (gmr_list != NULL) {
        { /* inlined gmr_destroy(gmr_list, &gmr_list->group) */
          gmr_t *mreg = gmr_list;
          ARMCI_Group *group = &gmr_list->group;
          int search_proc_in, search_proc_out, search_proc_out_grp;
          void *search_base = NULL;
          int alloc_me, alloc_nproc, world_me, world_nproc;
          MPI_Comm_rank(group->comm, &alloc_me);
          MPI_Comm_size(group->comm, &alloc_nproc);
          MPI_Comm_rank(ARMCI_GROUP_WORLD.comm, &world_me);
          MPI_Comm_size(ARMCI_GROUP_WORLD.comm, &world_nproc);
          if (mreg == NULL) search_proc_in = -1;
          else { search_proc_in = world_me; search_base = mreg->slices[world_me].base; }
          MPI_Allreduce(&search_proc_in, &search_proc_out, 1, MPI_INT, MPI_MAX, group->comm);
          if (search_proc_out >= 0) {
            search_proc_out_grp = search_proc_out;
            MPI_Bcast(&search_base, sizeof(void*), MPI_BYTE, search_proc_out_grp, group->comm);
            if (mreg == NULL) {
              void *lk_ptr = (search_base); int lk_proc = (search_proc_out);
              gmr_t *m = gmr_list;
              while (m != NULL) {
                if (lk_proc < m->nslices) {
                  const uint8_t   *base = m->slices[lk_proc].base;
                  const gmr_size_t sz   = m->slices[lk_proc].size;
                  if ((uint8_t*)lk_ptr >= base && (uint8_t*)lk_ptr < base + sz) break;
                }
                m = m->next;
              }
              mreg = m;
            }
            if (mreg->prev == NULL) {
              gmr_list = mreg->next;
              if (mreg->next != NULL) mreg->next->prev = NULL;
            } else {
              mreg->prev->next = mreg->next;
              if (mreg->next != NULL) mreg->next->prev = mreg->prev;
            }
            MPI_Win_unlock_all(window);
            MPI_Win_free(&window);
            free(mreg->slices);
            free(mreg);
          }
        }
        nfreed++;
      }
    
      if (nfreed > 0 && ARMCI_GROUP_WORLD.rank == 0) {
        ARMCII_Warning("Freed %d leaked allocations\n", nfreed);
      }
    
      { /* inlined ARMCI_Group_free(&ARMCI_GROUP_WORLD) */
        ARMCI_Group *group = &ARMCI_GROUP_WORLD;
        if (group->comm != MPI_COMM_NULL) {
          MPI_Comm_free(&group->comm);
        }
    
        if (group->abs_to_grp != NULL)
          free(group->abs_to_grp);
        if (group->grp_to_abs != NULL)
          free(group->grp_to_abs);
    
        group->rank = -1;
        group->size = 0;
      }
    }
    MPI_Finalize();
    return 0;
}
