

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <execinfo.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <unistd.h>


#define USE_RMA_REQUESTS 1

enum  ARMCI_Acc_e { ARMCI_ACC_INT  , ARMCI_ACC_LNG  ,
                    ARMCI_ACC_FLT  , ARMCI_ACC_DBL  ,
                    ARMCI_ACC_CPL  , ARMCI_ACC_DCP   };

typedef long armci_size_t;
#define ARMCII_MPI_SIZE_T MPI_LONG

int   ARMCI_Init(void);
int   ARMCI_Init_args(int *argc, char ***argv);
int   ARMCI_Init_thread(int armci_requested);
int   ARMCI_Init_thread_comm(int armci_requested, MPI_Comm comm);
int   ARMCI_Init_mpi_comm(MPI_Comm comm);
int   ARMCI_Initialized(void);

int   ARMCI_Finalize(void);
void  ARMCI_Cleanup(void);

void  ARMCI_Error(const char *msg, int code);

int   ARMCI_Malloc(void **base_ptrs, armci_size_t size);
int   ARMCI_Free(void *ptr);

void *ARMCI_Malloc_local(armci_size_t size);
int   ARMCI_Free_local(void *ptr);

void  ARMCI_Barrier(void);
void  ARMCI_Fence(int proc);
void  ARMCI_AllFence(void);

void  ARMCI_Access_begin(void *ptr);
void  ARMCI_Access_end(void *ptr);

void  ARMCI_Copy(void *src, void *dst, int size);

int   ARMCI_Get(void *src, void *dst, int size, int target);
int   ARMCI_Put(void *src, void *dst, int size, int target);
int   ARMCI_Acc(int datatype, void *scale, void *src, void *dst, int bytes, int proc);

int   ARMCI_PutS(void *src_ptr, int src_stride_ar[ ],
                 void *dst_ptr, int dst_stride_ar[ ],
                 int count[ ], int stride_levels, int proc);
int   ARMCI_GetS(void *src_ptr, int src_stride_ar[ ],
                 void *dst_ptr, int dst_stride_ar[ ],
                 int count[ ], int stride_levels, int proc);
int   ARMCI_AccS(int datatype, void *scale,
                 void *src_ptr, int src_stride_ar[ ],
                 void *dst_ptr, int dst_stride_ar[ ],
                 int count[ ], int stride_levels, int proc);

int   ARMCI_Put_flag(void *src, void* dst, int size, int *flag, int value, int proc);
int   ARMCI_PutS_flag(void *src_ptr, int src_stride_ar[ ],
                 void *dst_ptr, int dst_stride_ar[ ],
                 int count[ ], int stride_levels,
                 int *flag, int value, int proc);

typedef struct armci_hdl_s
{
    int batch_size;
    MPI_Request single_request;
    MPI_Request *request_array;
}
armci_hdl_t;

void  ARMCI_INIT_HANDLE(armci_hdl_t *hdl);
void  ARMCI_SET_AGGREGATE_HANDLE(armci_hdl_t* handle);
void  ARMCI_UNSET_AGGREGATE_HANDLE(armci_hdl_t* handle);

int   ARMCI_NbPut(void *src, void *dst, int bytes, int proc, armci_hdl_t *hdl);
int   ARMCI_NbGet(void *src, void *dst, int bytes, int proc, armci_hdl_t *hdl);
int   ARMCI_NbAcc(int datatype, void *scale, void *src, void *dst, int bytes, int proc, armci_hdl_t *hdl);

int   ARMCI_Wait(armci_hdl_t* hdl);
int   ARMCI_Test(armci_hdl_t* hdl);
int   ARMCI_WaitProc(int proc);
int   ARMCI_WaitAll(void);

int   ARMCI_NbPutS(void *src_ptr, int src_stride_ar[ ],
                   void *dst_ptr, int dst_stride_ar[ ],
                   int count[ ], int stride_levels, int proc, armci_hdl_t *hdl);
int   ARMCI_NbGetS(void *src_ptr, int src_stride_ar[ ],
                   void *dst_ptr, int dst_stride_ar[ ],
                   int count[ ], int stride_levels, int proc, armci_hdl_t *hdl);
int   ARMCI_NbAccS(int datatype, void *scale,
                   void *src_ptr, int src_stride_ar[ ],
                   void *dst_ptr, int dst_stride_ar[ ],
                   int count[ ], int stride_levels, int proc, armci_hdl_t *hdl);

void armci_write_strided(void *ptr, int stride_levels, int stride_arr[], int count[], char *buf);
void armci_read_strided(void *ptr, int stride_levels, int stride_arr[], int count[], char *buf);

typedef struct {
  void **src_ptr_array;
  void **dst_ptr_array;
  int    bytes;
  int    ptr_array_len;
} armci_giov_t;

int ARMCI_PutV(armci_giov_t *iov, int iov_len, int proc);
int ARMCI_GetV(armci_giov_t *iov, int iov_len, int proc);
int ARMCI_AccV(int datatype, void *scale, armci_giov_t *iov, int iov_len, int proc);

int ARMCI_NbPutV(armci_giov_t *iov, int iov_len, int proc, armci_hdl_t* handle);
int ARMCI_NbGetV(armci_giov_t *iov, int iov_len, int proc, armci_hdl_t* handle);
int ARMCI_NbAccV(int datatype, void *scale, armci_giov_t *iov, int iov_len, int proc, armci_hdl_t* handle);

int ARMCI_PutValueInt(int src, void *dst, int proc);
int ARMCI_PutValueLong(long src, void *dst, int proc);
int ARMCI_PutValueFloat(float src, void *dst, int proc);
int ARMCI_PutValueDouble(double src, void *dst, int proc);

int ARMCI_NbPutValueInt(int src, void *dst, int proc, armci_hdl_t *hdl);
int ARMCI_NbPutValueLong(long src, void *dst, int proc, armci_hdl_t *hdl);
int ARMCI_NbPutValueFloat(float src, void *dst, int proc, armci_hdl_t *hdl);
int ARMCI_NbPutValueDouble(double src, void *dst, int proc, armci_hdl_t *hdl);

int    ARMCI_GetValueInt(void *src, int proc);
long   ARMCI_GetValueLong(void *src, int proc);
float  ARMCI_GetValueFloat(void *src, int proc);
double ARMCI_GetValueDouble(void *src, int proc);

int   ARMCI_Create_mutexes(int count);
int   ARMCI_Destroy_mutexes(void);
void  ARMCI_Lock(int mutex, int proc);
void  ARMCI_Unlock(int mutex, int proc);

enum ARMCI_Rmw_e { ARMCI_FETCH_AND_ADD, ARMCI_FETCH_AND_ADD_LONG,
                   ARMCI_SWAP, ARMCI_SWAP_LONG };

int ARMCI_Rmw(int op, void *ploc, void *prem, int value, int proc);

typedef struct {
  MPI_Comm  comm;
  MPI_Comm  noncoll_pgroup_comm;
  int      *grp_to_abs;
  int      *abs_to_grp;
  int       rank;
  int       size;
} ARMCI_Group;

void ARMCI_Group_create(int grp_size, int *pid_list, ARMCI_Group *group_out);
void ARMCI_Group_create_child(int grp_size, int *pid_list, ARMCI_Group *group_out, ARMCI_Group *group_parent);
void ARMCI_Group_free(ARMCI_Group *group);

int  ARMCI_Group_rank(ARMCI_Group *group, int *rank);
void ARMCI_Group_size(ARMCI_Group *group, int *size);

void ARMCI_Group_set_default(ARMCI_Group *group);
void ARMCI_Group_get_default(ARMCI_Group *group_out);
void ARMCI_Group_get_world(ARMCI_Group *group_out);

int ARMCI_Absolute_id(ARMCI_Group *group,int group_rank);

int ARMCI_Malloc_group(void **ptr_arr, armci_size_t bytes, ARMCI_Group *group);
int ARMCI_Free_group(void *ptr, ARMCI_Group *group);

enum armci_domain_e { ARMCI_DOMAIN_SMP };

typedef int armci_domain_t;

int armci_domain_nprocs(armci_domain_t domain, int id);
int armci_domain_id(armci_domain_t domain, int glob_proc_id);
int armci_domain_glob_proc_id(armci_domain_t domain, int id, int loc_proc_id);
int armci_domain_my_id(armci_domain_t domain);
int armci_domain_count(armci_domain_t domain);
int armci_domain_same_id(armci_domain_t domain, int proc);

int ARMCI_Same_node(int proc);

int  ARMCI_Uses_shm(void);
void ARMCI_Set_shm_limit(unsigned long shmemlimit);
int  ARMCI_Uses_shm_grp(ARMCI_Group *group);

int     PARMCI_Init(void);
int     PARMCI_Init_args(int *argc, char ***argv);
int     PARMCI_Init_thread(int armci_requested);
int     PARMCI_Init_thread_comm(int armci_requested, MPI_Comm comm);
int     PARMCI_Init_mpi_comm(MPI_Comm comm);
int     PARMCI_Initialized(void);
int     PARMCI_Finalize(void);

int     PARMCI_Malloc(void **base_ptrs, armci_size_t size);
int     PARMCI_Free(void *ptr);
void   *PARMCI_Malloc_local(armci_size_t size);
int     PARMCI_Free_local(void *ptr);

void    PARMCI_Barrier(void);
void    PARMCI_Fence(int proc);
void    PARMCI_AllFence(void);
void    PARMCI_Access_begin(void *ptr);
void    PARMCI_Access_end(void *ptr);

int     PARMCI_Get(void *src, void *dst, int size, int target);
int     PARMCI_Put(void *src, void *dst, int size, int target);
int     PARMCI_Acc(int datatype, void *scale, void *src, void *dst, int bytes, int proc);

int     PARMCI_PutS(void *src_ptr, int src_stride_ar[], void *dst_ptr, int dst_stride_ar[],
                 int count[], int stride_levels, int proc);
int     PARMCI_GetS(void *src_ptr, int src_stride_ar[], void *dst_ptr, int dst_stride_ar[],
                 int count[], int stride_levels, int proc);
int     PARMCI_AccS(int datatype, void *scale, void *src_ptr, int src_stride_ar[],
                 void *dst_ptr, int dst_stride_ar[], int count[], int stride_levels, int proc);
int     PARMCI_Put_flag(void *src, void* dst, int size, int *flag, int value, int proc);
int     PARMCI_PutS_flag(void *src_ptr, int src_stride_ar[], void *dst_ptr, int dst_stride_ar[],
                 int count[], int stride_levels, int *flag, int value, int proc);

int     PARMCI_PutV(armci_giov_t *iov, int iov_len, int proc);
int     PARMCI_GetV(armci_giov_t *iov, int iov_len, int proc);
int     PARMCI_AccV(int datatype, void *scale, armci_giov_t *iov, int iov_len, int proc);

int     PARMCI_Wait(armci_hdl_t* hdl);
int     PARMCI_Test(armci_hdl_t* hdl);
int     PARMCI_WaitProc(int proc);
int     PARMCI_WaitAll(void);

int     PARMCI_NbPut(void *src, void *dst, int bytes, int proc, armci_hdl_t *hdl);
int     PARMCI_NbGet(void *src, void *dst, int bytes, int proc, armci_hdl_t *hdl);
int     PARMCI_NbAcc(int datatype, void *scale, void *src, void *dst, int bytes, int proc, armci_hdl_t *hdl);
int     PARMCI_NbPutS(void *src_ptr, int src_stride_ar[], void *dst_ptr, int dst_stride_ar[],
                   int count[], int stride_levels, int proc, armci_hdl_t *hdl);
int     PARMCI_NbGetS(void *src_ptr, int src_stride_ar[], void *dst_ptr, int dst_stride_ar[],
                   int count[], int stride_levels, int proc, armci_hdl_t *hdl);
int     PARMCI_NbAccS(int datatype, void *scale, void *src_ptr, int src_stride_ar[],
                   void *dst_ptr, int dst_stride_ar[], int count[], int stride_levels, int proc, armci_hdl_t *hdl);
int     PARMCI_NbPutV(armci_giov_t *iov, int iov_len, int proc, armci_hdl_t* handle);
int     PARMCI_NbGetV(armci_giov_t *iov, int iov_len, int proc, armci_hdl_t* handle);
int     PARMCI_NbAccV(int datatype, void *scale, armci_giov_t *iov, int iov_len, int proc, armci_hdl_t* handle);

int     PARMCI_PutValueInt(int src, void *dst, int proc);
int     PARMCI_PutValueLong(long src, void *dst, int proc);
int     PARMCI_PutValueFloat(float src, void *dst, int proc);
int     PARMCI_PutValueDouble(double src, void *dst, int proc);
int     PARMCI_NbPutValueInt(int src, void *dst, int proc, armci_hdl_t *hdl);
int     PARMCI_NbPutValueLong(long src, void *dst, int proc, armci_hdl_t *hdl);
int     PARMCI_NbPutValueFloat(float src, void *dst, int proc, armci_hdl_t *hdl);
int     PARMCI_NbPutValueDouble(double src, void *dst, int proc, armci_hdl_t *hdl);

int     PARMCI_GetValueInt(void *src, int proc);
long    PARMCI_GetValueLong(void *src, int proc);
float   PARMCI_GetValueFloat(void *src, int proc);
double  PARMCI_GetValueDouble(void *src, int proc);

int     PARMCI_Create_mutexes(int count);
int     PARMCI_Destroy_mutexes(void);
void    PARMCI_Lock(int mutex, int proc);
void    PARMCI_Unlock(int mutex, int proc);
int     PARMCI_Rmw(int op, void *ploc, void *prem, int value, int proc);

void    parmci_msg_barrier(void);
void    parmci_msg_group_barrier(ARMCI_Group *group);

int ARMCI_Malloc_memdev(void **ptr_arr, armci_size_t bytes, const char* device);
int ARMCI_Malloc_group_memdev(void **ptr_arr, armci_size_t bytes, ARMCI_Group *group, const char *device);
int ARMCI_Free_memdev(void *ptr);

int PARMCI_Malloc_memdev(void **ptr_arr, armci_size_t bytes, const char* device);
int PARMCI_Malloc_group_memdev(void **ptr_arr, armci_size_t bytes, ARMCI_Group *group, const char *device);
int PARMCI_Free_memdev(void *ptr);

#if ( defined(__GNUC__) && (__GNUC__ >= 3) ) || defined(__IBMC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#  define unlikely(x_) __builtin_expect(!!(x_),0)
#  define likely(x_)   __builtin_expect(!!(x_),1)
#else
#  define unlikely(x_) (x_)
#  define likely(x_)   (x_)
#endif

enum ARMCII_Op_e { ARMCII_OP_PUT, ARMCII_OP_GET, ARMCII_OP_ACC };

enum ARMCII_Strided_methods_e { ARMCII_STRIDED_IOV, ARMCII_STRIDED_DIRECT };

enum ARMCII_Iov_methods_e { ARMCII_IOV_AUTO, ARMCII_IOV_CONSRV,
                            ARMCII_IOV_BATCHED, ARMCII_IOV_DIRECT };

enum ARMCII_Shr_buf_methods_e { ARMCII_SHR_BUF_COPY, ARMCII_SHR_BUF_NOGUARD };

extern char ARMCII_Strided_methods_str[][10];
extern char ARMCII_Iov_methods_str[][10];
extern char ARMCII_Shr_buf_methods_str[][10];

typedef struct {
  int           init_count;
  int           debug_alloc;
  int           iov_checks;
  int           iov_batched_limit;
  int           iov_dtype_chunk;
  int           noncollective_groups;
  int           cache_rank_translation;
  int           verbose;
  int           thread_level;
  int           use_win_allocate;
  int           msg_barrier_syncs;
  int           explicit_nb_progress;
  int           use_alloc_shm;
  int           rma_atomicity;
  int           end_to_end_flush;
  int           rma_nocheck;
  int           disable_shm_accumulate;
  int           use_same_op;
  int           use_request_atomics;
  int           flush_request_atomics;
  char          rma_ordering[20];

  size_t        memory_limit;

  enum ARMCII_Strided_methods_e strided_method;
  enum ARMCII_Iov_methods_e     iov_method;
  enum ARMCII_Shr_buf_methods_e shr_buf_method;
} global_state_t;

extern ARMCI_Group    ARMCI_GROUP_WORLD;
extern ARMCI_Group    ARMCI_GROUP_DEFAULT;
extern global_state_t ARMCII_GLOBAL_STATE;

void  ARMCII_Bzero(void *buf, armci_size_t size);
char *ARMCII_Getenv(const char *varname);
int   ARMCII_Getenv_bool(const char *varname, int default_value);
int   ARMCII_Getenv_int(const char *varname, int default_value);
long  ARMCII_Getenv_long(const char *varname, long default_value);
void ARMCII_Getenv_char(char * output, const char *varname, const char *default_value, int length);

void ARMCII_Sync_local(void);

int  ARMCII_Translate_absolute_to_group(ARMCI_Group *group, int world_rank);
void ARMCII_Group_init_from_comm(ARMCI_Group *group);

typedef struct {

  void *src;
  void *dst;
  int   stride_levels;

  int  *base_ptr;
  int  *src_stride_ar;
  int  *dst_stride_ar;
  int  *count;

  int   was_contiguous;
  int  *idx;
} armcii_iov_iter_t;

void ARMCII_Acc_type_translate(int armci_datatype, MPI_Datatype *type, int *type_size);

int  ARMCII_Iov_check_overlap(void **ptrs, int count, int size);
int  ARMCII_Iov_check_same_allocation(void **ptrs, int count, int proc);

void ARMCII_Strided_to_iov(armci_giov_t *iov,
               void *src_ptr, int src_stride_ar[ ],
               void *dst_ptr, int dst_stride_ar[ ],
               int count[ ], int stride_levels);

void ARMCII_Strided_to_dtype(int stride_array[ ], int count[ ],
                             int stride_levels, MPI_Datatype old_type, MPI_Datatype *new_type);

int ARMCII_Iov_op_dispatch(enum ARMCII_Op_e op, void **src, void **dst, int count, int size,
    int datatype, int overlapping, int same_alloc, int proc, int blocking, armci_hdl_t * handle);

int ARMCII_Iov_op_batched(enum ARMCII_Op_e op, void **src, void **dst, int count, int elem_count,
    MPI_Datatype type, int proc, int consrv  , int blocking, armci_hdl_t * handle);
int ARMCII_Iov_op_datatype(enum ARMCII_Op_e op, void **src, void **dst, int count, int elem_count,
    MPI_Datatype type, int proc, int blocking, armci_hdl_t * handle);

armcii_iov_iter_t *ARMCII_Strided_to_iov_iter(
               void *src_ptr, int src_stride_ar[ ],
               void *dst_ptr, int dst_stride_ar[ ],
               int count[ ], int stride_levels);
void ARMCII_Iov_iter_free(armcii_iov_iter_t *it);
int  ARMCII_Iov_iter_has_next(armcii_iov_iter_t *it);
int  ARMCII_Iov_iter_next(armcii_iov_iter_t *it, void **src, void **dst);

int  ARMCII_Buf_prepare_read_vec(void **orig_bufs, void ***new_bufs_ptr, int count, int size);
void ARMCII_Buf_finish_read_vec(void **orig_bufs, void **new_bufs, int count, int size);
int  ARMCII_Buf_prepare_acc_vec(void **orig_bufs, void ***new_bufs_ptr, int count, int size,
                            int datatype, void *scale);
void ARMCII_Buf_finish_acc_vec(void **orig_bufs, void **new_bufs, int count, int size);
int  ARMCII_Buf_prepare_write_vec(void **orig_bufs, void ***new_bufs_ptr, int count, int size);
void ARMCII_Buf_finish_write_vec(void **orig_bufs, void **new_bufs, int count, int size);

int  ARMCII_Buf_acc_is_scaled(int datatype, void *scale);
void ARMCII_Buf_acc_scale(void *buf_in, void *buf_out, int size, int datatype, void *scale);

int ARMCII_Is_win_unified(MPI_Win win);
void ARMCII_Sync(void);

enum debug_cats_e {
  DEBUG_CAT_ALL        =  -1,
  DEBUG_CAT_NONE       =   0,
  DEBUG_CAT_MEM_REGION = 0x1,
  DEBUG_CAT_ALLOC      = 0x2,
  DEBUG_CAT_MUTEX      = 0x4,
  DEBUG_CAT_GROUPS     = 0x8,
  DEBUG_CAT_CTREE      = 0x10,
  DEBUG_CAT_IOV        = 0x20
};

extern  unsigned DEBUG_CATS_ENABLED;

void    ARMCII_Assert_fail(const char *expr, const char *msg, const char *file, int line, const char *func);
#define ARMCII_Assert(EXPR)          do { if (unlikely(!(EXPR))) ARMCII_Assert_fail(#EXPR, NULL, __FILE__, __LINE__, __func__); } while(0)
#define ARMCII_Assert_msg(EXPR, MSG) do { if (unlikely(!(EXPR))) ARMCII_Assert_fail(#EXPR, MSG,  __FILE__, __LINE__, __func__); } while(0)

#define DEBUG_CAT_ENABLED(X) (DEBUG_CATS_ENABLED & (X))
void    ARMCII_Dbg_print_impl(const char *func, const char *format, ...);
#define ARMCII_Dbg_print(CAT,...) do { if (DEBUG_CAT_ENABLED(CAT)) ARMCII_Dbg_print_impl(__func__,__VA_ARGS__); } while (0)

#define ARMCII_Error(...) ARMCII_Error_impl(__FILE__,__LINE__,__func__,__VA_ARGS__)
void    ARMCII_Error_impl(const char *file, const int line, const char *func, const char *msg, ...);
void    ARMCII_Warning(const char *fmt, ...);


enum armci_scope_e { SCOPE_ALL, SCOPE_NODE, SCOPE_MASTERS};

enum armci_type_e  { ARMCI_INT, ARMCI_LONG, ARMCI_LONG_LONG, ARMCI_FLOAT, ARMCI_DOUBLE };

int  armci_msg_me(void);
int  armci_msg_nproc(void);

void armci_msg_abort(int code);
double armci_timer(void);

void armci_msg_snd(int tag, void *buffer, int len, int to);
void armci_msg_rcv(int tag, void *buffer, int buflen, int *msglen, int from);
int  armci_msg_rcvany(int tag, void *buffer, int buflen, int *msglen);

void armci_msg_barrier(void);
void armci_msg_group_barrier(ARMCI_Group *group);
void armci_msg_bintree(int scope, int *Root, int *Up, int *Left, int *Right);

void armci_msg_bcast(void *buffer, int len, int root);
void armci_msg_bcast_scope(int scope, void *buffer, int len, int root);
void armci_msg_brdcst(void *buffer, int len, int root);
void armci_msg_group_bcast_scope(int scope, void *buf, int len, int root, ARMCI_Group *group);

  void armci_msg_reduce(void *x, int n, char *op, int type);
  void armci_msg_reduce_scope(int scope, void *x, int n, char *op, int type);

void armci_msg_sel(void *x, int n, char *op, int type, int contribute);
void armci_msg_sel_scope(int scope, void *x, int n, char *op, int type, int contribute);

  void armci_exchange_address(void *ptr_ar[], int n);

  void armci_msg_clus_brdcst(void *buf, int len);
  void armci_msg_clus_igop(int *x, int n, char *op);
  void armci_msg_clus_fgop(float *x, int n, char *op);
  void armci_msg_clus_lgop(long *x, int n, char *op);
  void armci_msg_clus_llgop(long long *x, int n, char *op);
  void armci_msg_clus_dgop(double *x, int n, char *op);

  void armci_exchange_address_grp(void *ptr_arr[], int n, ARMCI_Group *group);
  void armci_grp_clus_brdcst(void *buf, int len, int grp_master, int grp_clus_nproc,ARMCI_Group *mastergroup);

void armci_msg_gop_scope(int scope, void *x, int n, char *op, int type);
void armci_msg_igop(int *x, int n, char *op);
void armci_msg_lgop(long *x, int n, char *op);
void armci_msg_llgop(long long *x, int n, char *op);
void armci_msg_fgop(float *x, int n, char *op);
void armci_msg_dgop(double *x, int n, char *op);

void armci_msg_group_gop_scope(int scope, void *x, int n, char *op, int type, ARMCI_Group *group);
void armci_msg_group_igop(int *x, int n, char *op, ARMCI_Group *group);
void armci_msg_group_lgop(long *x, int n, char *op, ARMCI_Group *group);
void armci_msg_group_llgop(long long *x, int n, char *op, ARMCI_Group *group);
void armci_msg_group_fgop(float *x, int n, char *op, ARMCI_Group *group);
void armci_msg_group_dgop(double *x, int n,char *op, ARMCI_Group *group);

int ARMCIX_Group_split(ARMCI_Group *parent, int color, int key, ARMCI_Group *new_group);
int ARMCIX_Group_dup(ARMCI_Group *parent, ARMCI_Group *new_group);

struct armcix_mutex_hdl_s {
  int         my_count;
  int         max_count;
  ARMCI_Group grp;
  MPI_Win    *windows;
  uint8_t   **bases;
};

typedef struct armcix_mutex_hdl_s * armcix_mutex_hdl_t;

armcix_mutex_hdl_t ARMCIX_Create_mutexes_hdl(int count, ARMCI_Group *pgroup);
int  ARMCIX_Destroy_mutexes_hdl(armcix_mutex_hdl_t hdl);
void ARMCIX_Lock_hdl(armcix_mutex_hdl_t hdl, int mutex, int proc);
int  ARMCIX_Trylock_hdl(armcix_mutex_hdl_t hdl, int mutex, int proc);
void ARMCIX_Unlock_hdl(armcix_mutex_hdl_t hdl, int mutex, int proc);

void ARMCIX_Progress(void);

typedef armci_size_t gmr_size_t;
#define GMR_MPI_SIZE_T ARMCII_MPI_SIZE_T

typedef struct {
  void       *base;
  gmr_size_t  size;
} gmr_slice_t;

typedef struct gmr_s {
  MPI_Win                 window;
  ARMCI_Group             group;

  struct gmr_s           *prev;
  struct gmr_s           *next;
  gmr_slice_t            *slices;
  int                     nslices;
  bool                    unified;
} gmr_t;

extern gmr_t *gmr_list;

gmr_t *gmr_create(gmr_size_t local_size, void **base_ptrs, ARMCI_Group *group);
void   gmr_destroy(gmr_t *mreg, ARMCI_Group *group);
int    gmr_destroy_all(void);
gmr_t *gmr_lookup(void *ptr, int proc);

int gmr_fetch_and_op(gmr_t *mreg, void *src, void *out, void *dst, MPI_Datatype type, MPI_Op op, int proc);

int gmr_get(gmr_t *mreg, void *src, void *dst, int size,
            int target, armci_hdl_t * handle);
int gmr_put(gmr_t *mreg, void *src, void *dst, int size,
            int target, armci_hdl_t * handle);
int gmr_accumulate(gmr_t *mreg, void *src, void *dst, int count, MPI_Datatype type,
                   int proc, armci_hdl_t * handle);
int gmr_get_accumulate(gmr_t *mreg, void *src, void *out, void *dst, int count, MPI_Datatype type,
                       MPI_Op op, int proc, armci_hdl_t * handle);

int gmr_get_typed(gmr_t *mreg, void *src, int src_count, MPI_Datatype src_type,
                  void *dst, int dst_count, MPI_Datatype dst_type,
                  int proc, armci_hdl_t * handle);
int gmr_put_typed(gmr_t *mreg, void *src, int src_count, MPI_Datatype src_type,
                  void *dst, int dst_count, MPI_Datatype dst_type,
                  int proc, armci_hdl_t * handle);
int gmr_accumulate_typed(gmr_t *mreg, void *src, int src_count, MPI_Datatype src_type,
                         void *dst, int dst_count, MPI_Datatype dst_type,
                         int proc, armci_hdl_t * handle);
int gmr_get_accumulate_typed(gmr_t *mreg, void *src, int src_count, MPI_Datatype src_type,
                             void *out, int out_count, MPI_Datatype out_type,
                             void *dst, int dst_count, MPI_Datatype dst_type,
                             MPI_Op op, int proc, armci_hdl_t * handle);

int gmr_lockall(gmr_t *mreg);
int gmr_unlockall(gmr_t *mreg);
int gmr_flush(gmr_t *mreg, int proc, int local_only);
int gmr_flushall(gmr_t *mreg, int local_only);
int gmr_sync(gmr_t *mreg);
int gmr_wait(armci_hdl_t * handle);

void gmr_progress(void);
void gmr_handle_add_request(armci_hdl_t * handle, MPI_Request req);

global_state_t ARMCII_GLOBAL_STATE = { 0 };

char ARMCII_Strided_methods_str[][10] = { "IOV", "DIRECT" };
char ARMCII_Iov_methods_str[][10]     = { "AUTO", "CONSRV", "BATCHED", "DIRECT" };
char ARMCII_Shr_buf_methods_str[][10] = { "COPY", "NOGUARD" };

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

int ARMCII_Translate_absolute_to_group(ARMCI_Group *group, int world_rank) {
  int       group_rank;
  MPI_Group world_group, sub_group;

  ARMCII_Assert(world_rank >= 0 && world_rank < ARMCI_GROUP_WORLD.size);

  if (group->comm == ARMCI_GROUP_WORLD.comm) {
    group_rank = world_rank;
    return group_rank;
  }

  else if (group->grp_to_abs != NULL) {
    group_rank = group->abs_to_grp[world_rank];
    return group_rank;
  }
  else {

    MPI_Comm_group(ARMCI_GROUP_WORLD.comm, &world_group);
    MPI_Comm_group(group->comm, &sub_group);

    MPI_Group_translate_ranks(world_group, 1, &world_rank, sub_group, &group_rank);

    MPI_Group_free(&world_group);
    MPI_Group_free(&sub_group);
  }

  if (group_rank == MPI_UNDEFINED)
    return -1;
  else
    return group_rank;
}

void ARMCII_Acc_type_translate(int armci_datatype, MPI_Datatype *mpi_type, int *type_size) {

    switch (armci_datatype) {
      case ARMCI_ACC_INT:
        *mpi_type = MPI_INT;
        break;
      case ARMCI_ACC_LNG:
        *mpi_type = MPI_LONG;
        break;
      case ARMCI_ACC_FLT:
        *mpi_type = MPI_FLOAT;
        break;
      case ARMCI_ACC_DBL:
        *mpi_type = MPI_DOUBLE;
        break;
      case ARMCI_ACC_CPL:
        *mpi_type = MPI_FLOAT;
        break;
      case ARMCI_ACC_DCP:
        *mpi_type = MPI_DOUBLE;
        break;
      default:
        ARMCII_Error("unknown data type", 100);
        return;
    }

    MPI_Type_size(*mpi_type, type_size);
}

unsigned DEBUG_CATS_ENABLED =
    DEBUG_CAT_NONE;

void ARMCII_Assert_fail(const char *expr, const char *msg, const char *file, int line, const char *func) {
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (msg == NULL)
    fprintf(stderr, "[%d] ARMCI assert fail in %s() [%s:%d]: \"%s\"\n", rank, func, file, line, expr);
  else
    fprintf(stderr, "[%d] ARMCI assert fail in %s() [%s:%d]: \"%s\"\n"
                    "[%d] Message: \"%s\"\n", rank, func, file, line, expr, rank, msg);

  {

    const int SIZE = 100;
    int    j, nframes;
    void  *frames[SIZE];
    char **symbols;

    nframes = backtrace(frames, SIZE);
    symbols = backtrace_symbols(frames, nframes);

    if (symbols == NULL)
      perror("Backtrace failure");

    fprintf(stderr, "[%d] Backtrace:\n", rank);
    for (j = 0; j < nframes; j++)
      fprintf(stderr, "[%d]  %2d - %s\n", rank, nframes-j-1, symbols[j]);

    free(symbols);
  }

  fflush(NULL);
  {
    double stall = MPI_Wtime();
    while (MPI_Wtime() - stall < 1) ;
  }
  MPI_Abort(MPI_COMM_WORLD, -1);
}

void ARMCII_Dbg_print_impl(const char *func, const char *format, ...) {
  va_list etc;
  int  disp;
  char string[500];

  disp  = 0;
  disp += snprintf(string, 500, "[%d] %s: ", ARMCI_GROUP_WORLD.rank, func);
  va_start(etc, format);
  disp += vsnprintf(string+disp, 500-disp, format, etc);
  va_end(etc);

  fprintf(stderr, "%s", string);
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

void PARMCI_AllFence(void) {
  gmr_t *cur_mreg = gmr_list;

  while (cur_mreg) {
    gmr_flushall(cur_mreg, 0);
    cur_mreg = cur_mreg->next;
  }
  return;
}

void ARMCI_Copy(void *src, void *dst, int size) {
  memmove(dst, src, size);
}

void ARMCII_Bzero(void *buf, armci_size_t size) {
  memset(buf, 0, (size_t)size);
}

int ARMCII_Getenv_bool(const char *varname, int default_value) {
  const char *var = getenv(varname);

  if (var == NULL) {
    return default_value;
  }
  if (var[0] == 'T' || var[0] == 't' || var[0] == '1' || var[0] == 'y' || var[0] == 'Y') {
    return 1;
  } else {
    return 0;
  }
}

char *ARMCII_Getenv(const char *varname) {
  return getenv(varname);
}

int ARMCII_Getenv_int(const char *varname, int default_value) {
  const char *var = getenv(varname);
  if (var) {
    return atoi(var);
  } else {
    return default_value;
  }
}

long ARMCII_Getenv_long(const char *varname, long default_value) {
  const char *var = getenv(varname);
  if (var) {
    return atol(var);
  } else {
    return default_value;
  }
}

void ARMCII_Getenv_char(char * output, const char *varname, const char *default_value, int length) {
  const char *var = getenv(varname);
  if (var) {
    if (strlen(var) > length) {
      ARMCII_Warning("ARMCI_Getenv_char: %s = %s is too long (%d)\n", varname, var, length);
    }
    strncpy(output, var, length);
  } else {
    strncpy(output, default_value, length);
  }
}

int ARMCII_Is_win_unified(MPI_Win win)
{
  void    *attr_ptr;
  int      attr_flag;

  MPI_Win_get_attr(win, MPI_WIN_MODEL, &attr_ptr, &attr_flag);
  if (attr_flag) {
    int * attr_val = (int*)attr_ptr;
    if ( (*attr_val)==MPI_WIN_UNIFIED ) {
      return 1;
    } else if ( (*attr_val)==MPI_WIN_UNIFIED ) {
      return 0;
    } else {
      return -1;
    }
  } else {
    return -1;
  }
}

#define MAX(A,B) (((A) > (B)) ? A : B)

#define MIN(X,Y) (((X) < (Y)) ? X : Y)
#define MAX(X,Y) (((X) > (Y)) ? X : Y)


ARMCI_Group ARMCI_GROUP_WORLD   = {0};
ARMCI_Group ARMCI_GROUP_DEFAULT = {0};

void ARMCII_Group_init_from_comm(ARMCI_Group *group) {
  if (group->comm != MPI_COMM_NULL) {
    MPI_Comm_size(group->comm, &group->size);
    MPI_Comm_rank(group->comm, &group->rank);

  } else {
    group->rank = -1;
    group->size =  0;
  }

  if (ARMCII_GLOBAL_STATE.noncollective_groups && group->comm != MPI_COMM_NULL)
    MPI_Comm_dup(group->comm, &group->noncoll_pgroup_comm);
  else
    group->noncoll_pgroup_comm = MPI_COMM_NULL;

  if (ARMCII_GLOBAL_STATE.cache_rank_translation) {
    if (group->comm != MPI_COMM_NULL) {
      int      *ranks, i;
      MPI_Group world_group, sub_group;

      group->abs_to_grp = malloc(sizeof(int)*ARMCI_GROUP_WORLD.size);
      group->grp_to_abs = malloc(sizeof(int)*group->size);
      ranks = malloc(sizeof(int)*ARMCI_GROUP_WORLD.size);

      ARMCII_Assert(group->abs_to_grp != NULL && group->grp_to_abs != NULL && ranks != NULL);

      for (i = 0; i < ARMCI_GROUP_WORLD.size; i++)
        ranks[i] = i;

      MPI_Comm_group(ARMCI_GROUP_WORLD.comm, &world_group);
      MPI_Comm_group(group->comm, &sub_group);

      MPI_Group_translate_ranks(sub_group, group->size, ranks, world_group, group->grp_to_abs);
      MPI_Group_translate_ranks(world_group, ARMCI_GROUP_WORLD.size, ranks, sub_group, group->abs_to_grp);

      MPI_Group_free(&world_group);
      MPI_Group_free(&sub_group);

      free(ranks);
    }
  }

  else {
    group->abs_to_grp = NULL;
    group->grp_to_abs = NULL;
  }
}

void ARMCI_Group_free(ARMCI_Group *group) {
  if (group->comm != MPI_COMM_NULL) {
    MPI_Comm_free(&group->comm);

    if (ARMCII_GLOBAL_STATE.noncollective_groups)
      MPI_Comm_free(&group->noncoll_pgroup_comm);
  }

  if (group->abs_to_grp != NULL)
    free(group->abs_to_grp);
  if (group->grp_to_abs != NULL)
    free(group->grp_to_abs);

  group->rank = -1;
  group->size = 0;
}

int PARMCI_Malloc(void **ptr_arr, armci_size_t bytes) {
  return ARMCI_Malloc_group(ptr_arr, bytes, &ARMCI_GROUP_WORLD);
}

int PARMCI_Free(void *ptr) {
  return ARMCI_Free_group(ptr, &ARMCI_GROUP_WORLD);
}

int ARMCI_Malloc_group(void **base_ptrs, armci_size_t size, ARMCI_Group *group) {
  int i;
  gmr_t *mreg;

  ARMCII_Assert(PARMCI_Initialized());

  mreg = gmr_create(size, base_ptrs, group);

  if (DEBUG_CAT_ENABLED(DEBUG_CAT_ALLOC)) {
#define BUF_LEN 1000
    char ptr_string[BUF_LEN];
    int  count = 0;

    if (mreg == NULL) {
      strncpy(ptr_string, "NULL", 5);
    } else {
      for (i = 0; i < mreg->nslices && count < BUF_LEN; i++)
        count += snprintf(ptr_string+count, BUF_LEN-count,
            (i == mreg->nslices-1) ? "%p" : "%p ", base_ptrs[i]);
    }

    ARMCII_Dbg_print(DEBUG_CAT_ALLOC, "base ptrs [%s]\n", ptr_string);
#undef BUF_LEN
  }

  return 0;
}

int ARMCI_Free_group(void *ptr, ARMCI_Group *group) {
  gmr_t *mreg;

  if (ptr != NULL) {
    mreg = gmr_lookup(ptr, ARMCI_GROUP_WORLD.rank);
    ARMCII_Assert_msg(mreg != NULL, "Invalid shared pointer");
  } else {
    ARMCII_Dbg_print(DEBUG_CAT_ALLOC, "given NULL\n");
    mreg = NULL;
  }
  gmr_destroy(mreg, group);

  return 0;
}

gmr_t *gmr_list = NULL;

gmr_t *gmr_create(gmr_size_t local_size, void **base_ptrs, ARMCI_Group *group) {
  int           i;
  int           alloc_me, alloc_nproc;
  int           world_me, world_nproc;
  MPI_Group     world_group, alloc_group;
  gmr_t        *mreg;
  gmr_slice_t  *alloc_slices, gmr_slice;

  ARMCII_Assert(local_size >= 0);
  ARMCII_Assert(group != NULL);

  MPI_Comm_rank(group->comm, &alloc_me);
  MPI_Comm_size(group->comm, &alloc_nproc);

  gmr_size_t max_local_size;
  {

    MPI_Allreduce(&local_size, &max_local_size, 1, GMR_MPI_SIZE_T, MPI_MAX, group->comm);

    if (max_local_size==0) {
      for (i = 0; i < alloc_nproc; i++) {
        base_ptrs[i] = NULL;
      }
      return NULL;
    }
  }

  MPI_Comm_rank(ARMCI_GROUP_WORLD.comm, &world_me);
  MPI_Comm_size(ARMCI_GROUP_WORLD.comm, &world_nproc);

  mreg = malloc(sizeof(gmr_t));
  ARMCII_Assert(mreg != NULL);

  mreg->slices = malloc(sizeof(gmr_slice_t)*world_nproc);
  ARMCII_Assert(mreg->slices != NULL);
  alloc_slices = malloc(sizeof(gmr_slice_t)*alloc_nproc);
  ARMCII_Assert(alloc_slices != NULL);

  mreg->group          = *group;

  mreg->nslices        = world_nproc;
  mreg->prev           = NULL;
  mreg->next           = NULL;
  mreg->unified        = false;

  alloc_slices[alloc_me].size = local_size;

  MPI_Info win_info = MPI_INFO_NULL;
  MPI_Info_create(&win_info);

  if (ARMCII_GLOBAL_STATE.use_alloc_shm) {
      MPI_Info_set(win_info, "alloc_shm", "true");
  } else {
      MPI_Info_set(win_info, "alloc_shm", "false");
  }

  MPI_Info_set(win_info, "same_disp_unit", "true");

  MPI_Info_set(win_info, "alloc_shared_noncontig", "false");

  if ((ARMCII_GLOBAL_STATE.iov_method == ARMCII_IOV_CONSRV ||
       ARMCII_GLOBAL_STATE.iov_method == ARMCII_IOV_BATCHED) &&
      (ARMCII_GLOBAL_STATE.strided_method == ARMCII_STRIDED_IOV)) {
      MPI_Info_set(win_info, "accumulate_noncontig_dtype", "false");
  }

  {
      if (max_local_size > 2147483647) max_local_size = -1;
      char max_local_size_string[16] = {0};
      snprintf(max_local_size_string,sizeof(max_local_size_string)-1,"%ld",max_local_size);
      MPI_Info_set(win_info, "accumulate_max_bytes", max_local_size_string);
  }

  if (ARMCII_GLOBAL_STATE.disable_shm_accumulate) {
      MPI_Info_set(win_info, "disable_shm_accumulate", "true");
  }

  if (ARMCII_GLOBAL_STATE.use_same_op) {
      MPI_Info_set(win_info, "accumulate_ops", "same_op");
  }

  if (strlen(ARMCII_GLOBAL_STATE.rma_ordering) > 0) {
      MPI_Info_set(win_info, "accumulate_ordering", ARMCII_GLOBAL_STATE.rma_ordering);
  }

  MPI_Info_set(win_info, "epochs_used", "lockall");

  if (ARMCII_GLOBAL_STATE.use_win_allocate == 0) {

      if (local_size == 0) {
        alloc_slices[alloc_me].base = NULL;
      } else {
        MPI_Alloc_mem(local_size, win_info, &(alloc_slices[alloc_me].base));
        ARMCII_Assert(alloc_slices[alloc_me].base != NULL);
      }
      MPI_Win_create(alloc_slices[alloc_me].base, (MPI_Aint) local_size, 1, MPI_INFO_NULL, group->comm, &mreg->window);
  }
  else if (ARMCII_GLOBAL_STATE.use_win_allocate == 1) {

      MPI_Win_allocate( (MPI_Aint) local_size, 1, win_info, group->comm, &(alloc_slices[alloc_me].base), &mreg->window);

      if (local_size == 0) {

        alloc_slices[alloc_me].base = NULL;
      } else {
        ARMCII_Assert(alloc_slices[alloc_me].base != NULL);
      }
  }
  else {
      ARMCII_Error("invalid window type!\n");
  }

  MPI_Info_free(&win_info);

  if (ARMCII_GLOBAL_STATE.debug_alloc && local_size > 0) {
    ARMCII_Bzero(alloc_slices[alloc_me].base, local_size);
  }

  gmr_slice = alloc_slices[alloc_me];
  MPI_Allgather(  &gmr_slice, sizeof(gmr_slice_t), MPI_BYTE,
                 alloc_slices, sizeof(gmr_slice_t), MPI_BYTE, group->comm);

  for (i = 0; i < alloc_nproc; i++)
    base_ptrs[i] = alloc_slices[i].base;

  memset(mreg->slices, 0, sizeof(gmr_slice_t)*world_nproc);

  MPI_Comm_group(ARMCI_GROUP_WORLD.comm, &world_group);
  MPI_Comm_group(group->comm, &alloc_group);

  for (i = 0; i < alloc_nproc; i++) {
    int world_rank;
    MPI_Group_translate_ranks(alloc_group, 1, &i, world_group, &world_rank);
    mreg->slices[world_rank] = alloc_slices[i];
  }

  free(alloc_slices);
  MPI_Group_free(&world_group);
  MPI_Group_free(&alloc_group);

  MPI_Win_lock_all((ARMCII_GLOBAL_STATE.rma_nocheck) ? MPI_MODE_NOCHECK : 0,
                   mreg->window);

  {
    const int unified = ARMCII_Is_win_unified(mreg->window);
    const int print = ARMCII_GLOBAL_STATE.verbose;
    if (unified == 1) {
        mreg->unified = true;
        if (print > 1) printf("MPI_WIN_MODEL = MPI_WIN_UNIFIED\n");
    } else if (unified == 0) {
        mreg->unified = false;
        if (print > 1) printf("MPI_WIN_MODEL = MPI_WIN_SEPARATE\n");
    } else {
        mreg->unified = false;
        if (print > 1) printf("MPI_WIN_MODEL not available\n");
    }
    if (!(mreg->unified) && (ARMCII_GLOBAL_STATE.shr_buf_method == ARMCII_SHR_BUF_NOGUARD) ) {
      if (world_me==0) {
        printf("Please re-run with ARMCI_SHR_BUF_METHOD=COPY\n");
      }

    }
  }

  if (gmr_list == NULL) {
    gmr_list = mreg;

  } else {
    gmr_t *parent = gmr_list;

    while (parent->next != NULL)
      parent = parent->next;

    parent->next = mreg;
    mreg->prev   = parent;
  }

  return mreg;
}

void gmr_destroy(gmr_t *mreg, ARMCI_Group *group) {
  int   search_proc_in, search_proc_out, search_proc_out_grp;
  void *search_base = NULL;
  int   alloc_me, alloc_nproc;
  int   world_me, world_nproc;

  MPI_Comm_rank(group->comm, &alloc_me);
  MPI_Comm_size(group->comm, &alloc_nproc);
  MPI_Comm_rank(ARMCI_GROUP_WORLD.comm, &world_me);
  MPI_Comm_size(ARMCI_GROUP_WORLD.comm, &world_nproc);

  if (mreg == NULL)
    search_proc_in = -1;
  else {
    search_proc_in = world_me;
    search_base    = mreg->slices[world_me].base;
  }

  MPI_Allreduce(&search_proc_in, &search_proc_out, 1, MPI_INT, MPI_MAX, group->comm);

  if (search_proc_out < 0)
    return;

  search_proc_out_grp = ARMCII_Translate_absolute_to_group(group, search_proc_out);

  MPI_Bcast(&search_base, sizeof(void*), MPI_BYTE, search_proc_out_grp, group->comm);

  if (mreg == NULL)
    mreg = gmr_lookup(search_base, search_proc_out);

  ARMCII_Assert_msg(mreg != NULL, "Could not locate the desired allocation");

  if (mreg->prev == NULL) {
    ARMCII_Assert(gmr_list == mreg);
    gmr_list = mreg->next;

    if (mreg->next != NULL)
      mreg->next->prev = NULL;

  } else {
    mreg->prev->next = mreg->next;
    if (mreg->next != NULL)
      mreg->next->prev = mreg->prev;
  }

  ARMCII_Assert_msg(mreg->window != MPI_WIN_NULL, "A non-null mreg contains a null window.");
  MPI_Win_unlock_all(mreg->window);

  MPI_Win_free(&mreg->window);

  if (ARMCII_GLOBAL_STATE.use_win_allocate == 0) {
    if (mreg->slices[world_me].base != NULL) {
      MPI_Free_mem(mreg->slices[world_me].base);
    }
  }

  free(mreg->slices);
  free(mreg);
}

int gmr_destroy_all(void) {
  int count = 0;

  while (gmr_list != NULL) {
    gmr_destroy(gmr_list, &gmr_list->group);
    count++;
  }

  return count;
}

gmr_t *gmr_lookup(void *ptr, int proc) {
  gmr_t *mreg;

  mreg = gmr_list;

  while (mreg != NULL) {
    ARMCII_Assert(proc < mreg->nslices);

    if (proc < mreg->nslices) {
      const uint8_t   *base = mreg->slices[proc].base;
      const gmr_size_t size = mreg->slices[proc].size;

      if ((uint8_t*) ptr >= base && (uint8_t*) ptr < base + size)
        break;
    }

    mreg = mreg->next;
  }

  return mreg;
}

int gmr_put_typed(gmr_t *mreg, void *src, int src_count, MPI_Datatype src_type,
                  void *dst, int dst_count, MPI_Datatype dst_type,
                  int proc, armci_hdl_t * handle)
{
  int        grp_proc;
  gmr_size_t disp;
  MPI_Aint lb, extent;

  grp_proc = ARMCII_Translate_absolute_to_group(&mreg->group, proc);
  ARMCII_Assert(grp_proc >= 0);
  ARMCII_Assert_msg(mreg->window != MPI_WIN_NULL, "A non-null mreg contains a null window.");

  if (dst == MPI_BOTTOM) {
    disp = 0;
  } else {
    disp = (gmr_size_t) ((uint8_t*)dst - (uint8_t*)mreg->slices[proc].base);
  }

  MPI_Type_get_true_extent(dst_type, &lb, &extent);
  ARMCII_Assert_msg(disp >= 0 && disp < mreg->slices[proc].size, "Invalid remote address");
  ARMCII_Assert_msg(disp + dst_count*extent <= mreg->slices[proc].size, "Transfer is out of range");

  if (handle!=NULL) {

    MPI_Request req = MPI_REQUEST_NULL;

    if (ARMCII_GLOBAL_STATE.rma_atomicity) {
        MPI_Raccumulate(src, src_count, src_type, grp_proc,
                        (MPI_Aint) disp, dst_count, dst_type,
                        MPI_REPLACE, mreg->window, &req);
    } else {
        MPI_Rput(src, src_count, src_type, grp_proc,
                 (MPI_Aint) disp, dst_count, dst_type,
                 mreg->window, &req);
    }

    gmr_handle_add_request(handle, req);

    return 0;

  }

  if (ARMCII_GLOBAL_STATE.rma_atomicity) {
      MPI_Accumulate(src, src_count, src_type, grp_proc,
                     (MPI_Aint) disp, dst_count, dst_type,
                     MPI_REPLACE, mreg->window);
  } else {
      MPI_Put(src, src_count, src_type, grp_proc,
              (MPI_Aint) disp, dst_count, dst_type, mreg->window);
  }

  return 0;
}

int gmr_get(gmr_t *mreg, void *src, void *dst, int size, int proc, armci_hdl_t * handle)
{
  ARMCII_Assert_msg(dst != NULL, "Invalid local address");
  return gmr_get_typed(mreg, src, size, MPI_BYTE, dst, size, MPI_BYTE, proc, handle);
}

int gmr_get_typed(gmr_t *mreg, void *src, int src_count, MPI_Datatype src_type,
                  void *dst, int dst_count, MPI_Datatype dst_type,
                  int proc, armci_hdl_t * handle)
{
  int        grp_proc;
  gmr_size_t disp;
  MPI_Aint lb, extent;

  grp_proc = ARMCII_Translate_absolute_to_group(&mreg->group, proc);
  ARMCII_Assert(grp_proc >= 0);
  ARMCII_Assert_msg(mreg->window != MPI_WIN_NULL, "A non-null mreg contains a null window.");

  if (src == MPI_BOTTOM) {
    disp = 0;
  } else {
    disp = (gmr_size_t) ((uint8_t*)src - (uint8_t*)mreg->slices[proc].base);
  }

  MPI_Type_get_true_extent(src_type, &lb, &extent);
  ARMCII_Assert_msg(disp >= 0 && disp < mreg->slices[proc].size, "Invalid remote address");
  ARMCII_Assert_msg(disp + src_count*extent <= mreg->slices[proc].size, "Transfer is out of range");

  if (handle!=NULL) {

    MPI_Request req = MPI_REQUEST_NULL;

    if (ARMCII_GLOBAL_STATE.rma_atomicity) {

        MPI_Rget_accumulate(NULL, 0, src_type  ,
                            dst, dst_count, dst_type, grp_proc,
                            (MPI_Aint) disp, src_count, src_type,
                            MPI_NO_OP, mreg->window, &req);
    } else {
        MPI_Rget(dst, dst_count, dst_type, grp_proc,
                 (MPI_Aint) disp, src_count, src_type,
                 mreg->window, &req);
    }

    gmr_handle_add_request(handle, req);

    return 0;
  }

  if (ARMCII_GLOBAL_STATE.rma_atomicity) {
      MPI_Get_accumulate(NULL, 0, MPI_BYTE, dst, dst_count, dst_type, grp_proc,
                         (MPI_Aint) disp, src_count, src_type, MPI_NO_OP, mreg->window);
  } else {
      MPI_Get(dst, dst_count, dst_type, grp_proc,
              (MPI_Aint) disp, src_count, src_type, mreg->window);
  }

  return 0;
}

int gmr_accumulate_typed(gmr_t *mreg, void *src, int src_count, MPI_Datatype src_type,
                         void *dst, int dst_count, MPI_Datatype dst_type,
                         int proc, armci_hdl_t * handle)
{
  int        grp_proc;
  gmr_size_t disp;
  MPI_Aint lb, extent;

  grp_proc = ARMCII_Translate_absolute_to_group(&mreg->group, proc);
  ARMCII_Assert(grp_proc >= 0);
  ARMCII_Assert_msg(mreg->window != MPI_WIN_NULL, "A non-null mreg contains a null window.");

  if (dst == MPI_BOTTOM) {
    disp = 0;
  } else {
    disp = (gmr_size_t) ((uint8_t*)dst - (uint8_t*)mreg->slices[proc].base);
  }

  MPI_Type_get_true_extent(dst_type, &lb, &extent);
  ARMCII_Assert_msg(disp >= 0 && disp < mreg->slices[proc].size, "Invalid remote address");
  ARMCII_Assert_msg(disp + dst_count*extent <= mreg->slices[proc].size, "Transfer is out of range");

  if (handle!=NULL) {

    MPI_Request req = MPI_REQUEST_NULL;

    MPI_Raccumulate(src, src_count, src_type, grp_proc,
                    (MPI_Aint) disp, dst_count, dst_type,
                    MPI_SUM, mreg->window, &req);

    gmr_handle_add_request(handle, req);

    return 0;

  }

  MPI_Accumulate(src, src_count, src_type, grp_proc, (MPI_Aint) disp, dst_count, dst_type, MPI_SUM, mreg->window);

  return 0;
}

int gmr_flush(gmr_t *mreg, int proc, int local_only) {
  int grp_proc = ARMCII_Translate_absolute_to_group(&mreg->group, proc);
  int grp_me   = ARMCII_Translate_absolute_to_group(&mreg->group, ARMCI_GROUP_WORLD.rank);

  ARMCII_Assert(grp_proc >= 0 && grp_me >= 0);
  ARMCII_Assert_msg(mreg->window != MPI_WIN_NULL, "A non-null mreg contains a null window.");
  ARMCII_Assert_msg(grp_proc < mreg->group.size, "grp_proc exceeds group size!");

  if (!local_only || ARMCII_GLOBAL_STATE.end_to_end_flush) {
    MPI_Win_flush(grp_proc, mreg->window);
  } else {
    MPI_Win_flush_local(grp_proc, mreg->window);
  }

  return 0;
}

int gmr_flushall(gmr_t *mreg, int local_only) {
  int grp_me   = ARMCII_Translate_absolute_to_group(&mreg->group, ARMCI_GROUP_WORLD.rank);

  ARMCII_Assert(grp_me >= 0);
  ARMCII_Assert_msg(mreg->window != MPI_WIN_NULL, "A non-null mreg contains a null window.");

  if (!local_only || ARMCII_GLOBAL_STATE.end_to_end_flush) {
    MPI_Win_flush_all(mreg->window);
  } else {
    MPI_Win_flush_local_all(mreg->window);
  }

  return 0;
}

void gmr_handle_add_request(armci_hdl_t * handle, MPI_Request req)
{
  if (handle->batch_size < 0) {

    ARMCII_Warning("gmr_handle_add_request passed a bogus (uninitialized) handle.\n");

  } else if (handle->batch_size == 0) {

    if (handle->single_request != MPI_REQUEST_NULL) {
      ARMCII_Warning("gmr_handle_add_request: handle is corrupt (single_request_array is not MPI_REQUEST_NULL).\n");
    }
    if (handle->request_array != NULL) {

    }

    handle->batch_size     = 1;
    handle->single_request = req;

  } else if (handle->batch_size == 1) {

    if (handle->single_request == MPI_REQUEST_NULL) {
      ARMCII_Warning("gmr_handle_add_request: handle is corrupt (single_request_array is MPI_REQUEST_NULL).\n");
    }
    if (handle->request_array != NULL) {

    }

    handle->batch_size++;
    handle->request_array    = malloc( handle->batch_size * sizeof(MPI_Request) );
    handle->request_array[0] = handle->single_request;
    handle->request_array[1] = req;
    handle->single_request   = MPI_REQUEST_NULL;

  } else if (handle->batch_size > 1) {

    if (handle->single_request != MPI_REQUEST_NULL) {
      ARMCII_Warning("gmr_handle_add_request: handle is corrupt (single_request_array is not MPI_REQUEST_NULL).\n");
    }
    if (handle->request_array == NULL) {
      ARMCII_Warning("gmr_handle_add_request: handle is corrupt (request_array is NULL).\n");
    }

    handle->batch_size++;
    handle->request_array  = realloc( handle->request_array , handle->batch_size * sizeof(MPI_Request) );
    handle->request_array[handle->batch_size-1] = req;

  }
}

int PARMCI_Get(void *src, void *dst, int size, int target) {
  gmr_t *src_mreg, *dst_mreg;

  src_mreg = gmr_lookup(src, target);

  if (ARMCII_GLOBAL_STATE.shr_buf_method != ARMCII_SHR_BUF_NOGUARD)
    dst_mreg = gmr_lookup(dst, ARMCI_GROUP_WORLD.rank);
  else
    dst_mreg = NULL;

  ARMCII_Assert_msg(src_mreg != NULL, "Invalid remote pointer");

  if (target == ARMCI_GROUP_WORLD.rank && dst_mreg == NULL) {
    ARMCI_Copy(src, dst, size);
  }

  else if (dst_mreg == NULL) {
    gmr_get(src_mreg, src, dst, size, target, NULL  );
    gmr_flush(src_mreg, target, 0);
  }

  else {
    void *dst_buf;

    MPI_Alloc_mem(size, MPI_INFO_NULL, &dst_buf);
    ARMCII_Assert(dst_buf != NULL);

    gmr_get(src_mreg, src, dst_buf, size, target, NULL  );
    gmr_flush(src_mreg, target, 0);

    ARMCI_Copy(dst_buf, dst, size);

    MPI_Free_mem(dst_buf);
  }

  return 0;
}

int ARMCII_Iov_op_dispatch(enum ARMCII_Op_e op, void **src, void **dst, int count, int size,
                           int datatype, int overlapping, int same_alloc, int proc,
                           int blocking, armci_hdl_t * handle)
{

  MPI_Datatype type;
  int type_count, type_size;

  if (op == ARMCII_OP_ACC) {
    ARMCII_Acc_type_translate(datatype, &type, &type_size);
  } else {
    type = MPI_BYTE;
    MPI_Type_size(type, &type_size);
  }
  type_count = size/type_size;
  ARMCII_Assert_msg(size % type_size == 0, "Transfer size is not a multiple of type size");

  return ARMCII_Iov_op_datatype(op, src, dst, count, type_count, type, proc, blocking, handle);
}

int ARMCII_Iov_op_datatype(enum ARMCII_Op_e op, void **src, void **dst, int count, int elem_count,
                           MPI_Datatype type, int proc, int blocking, armci_hdl_t * handle)
{

    gmr_t *mreg;
    MPI_Datatype  type_loc, type_rem;
    int           disp_loc[count];
    int           disp_rem[count];
    int           block_len[count];
    MPI_Aint      loc_addr[count];
    MPI_Aint      base_loc;
    void         *base_loc_ptr;
    void         *dst_win_base;
    int           dst_win_size, i, type_size;
    void        **buf_rem, **buf_loc;
    MPI_Aint      base_rem;
    int flush_local = 0;

    switch(op) {
      case ARMCII_OP_ACC:
      case ARMCII_OP_PUT:
        buf_rem = dst;
        buf_loc = src;
        break;
      case ARMCII_OP_GET:
        buf_rem = src;
        buf_loc = dst;
        break;
      default:
        ARMCII_Error("unknown operation (%d)", op);
        return 1;
    }

    MPI_Type_size(type, &type_size);

    mreg = gmr_lookup(buf_rem[0], proc);
    ARMCII_Assert_msg(mreg != NULL, "Invalid remote pointer");

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
      block_len[i] = elem_count;

      ARMCII_Assert_msg((loc_addr[i] - base_loc) % type_size == 0, "Local transfer offset is not a multiple of type size");
      ARMCII_Assert_msg(off_loc <= INT_MAX, "Local segment span exceeds 32-bit element displacement; use ARMCI_IOV_METHOD=BATCHED");
      ARMCII_Assert_msg((target_rem - base_rem) % type_size == 0, "Transfer size is not a multiple of type size");
      ARMCII_Assert_msg(disp_rem[i] >= 0 && disp_rem[i] < dst_win_size, "Invalid remote pointer");
      ARMCII_Assert_msg(((uint8_t*)buf_rem[i]) + block_len[i] <= ((uint8_t*)dst_win_base) + dst_win_size, "Transfer exceeds buffer length");
      disp_loc[i]  = (int)off_loc;
    }

    int chunk = ARMCII_GLOBAL_STATE.iov_dtype_chunk;
    if (chunk <= 0 || chunk > count) chunk = count;

    for (int start = 0; start < count; start += chunk) {
      int n = (count - start < chunk) ? (count - start) : chunk;

      MPI_Type_create_indexed_block(n, elem_count, &disp_loc[start], type, &type_loc);
      MPI_Type_create_indexed_block(n, elem_count, &disp_rem[start], type, &type_rem);
      MPI_Type_commit(&type_loc);
      MPI_Type_commit(&type_rem);

      switch(op) {
        case ARMCII_OP_PUT:
          gmr_put_typed(mreg, base_loc_ptr, 1, type_loc, MPI_BOTTOM, 1, type_rem, proc, handle);
          flush_local = 1;
          break;
        case ARMCII_OP_GET:
          gmr_get_typed(mreg, MPI_BOTTOM, 1, type_rem, base_loc_ptr, 1, type_loc, proc, handle);
          flush_local = 0;
          break;
        case ARMCII_OP_ACC:
          gmr_accumulate_typed(mreg, base_loc_ptr, 1, type_loc, MPI_BOTTOM, 1, type_rem, proc, handle);
          flush_local = 1;
          break;
        default:
          ARMCII_Error("unknown operation (%d)", op);
          return 1;
      }

      MPI_Type_free(&type_loc);
      MPI_Type_free(&type_rem);
    }

    if (blocking) {
      gmr_flush(mreg, proc, flush_local);
    }

    return 0;
}

int PARMCI_AccV(int datatype, void *scale, armci_giov_t *iov, int iov_len, int proc)
{
  for (int v = 0; v < iov_len; v++) {
    void **src_buf;
    int    overlapping, same_alloc;

    if (iov[v].ptr_array_len == 0) continue;
    if (iov[v].bytes == 0) continue;

    overlapping = 0; same_alloc = 1;   /* IOV_CHECKS disabled */

    ARMCII_Buf_prepare_acc_vec(iov[v].src_ptr_array, &src_buf, iov[v].ptr_array_len, iov[v].bytes, datatype, scale);
    ARMCII_Iov_op_dispatch(ARMCII_OP_ACC, src_buf, iov[v].dst_ptr_array, iov[v].ptr_array_len, iov[v].bytes, datatype,
                           overlapping, same_alloc, proc, 1  , NULL);
    ARMCII_Buf_finish_acc_vec(iov[v].src_ptr_array, src_buf, iov[v].ptr_array_len, iov[v].bytes);
  }

  return 0;
}

int ARMCII_Buf_prepare_acc_vec(void **orig_bufs, void ***new_bufs_ptr, int count, int size,
                            int datatype, void *scale) {

  void **new_bufs;
  int i, scaled, num_moved = 0;

  new_bufs = malloc((count+1)*sizeof(void*));
  ARMCII_Assert(new_bufs != NULL);
  new_bufs[count] = NULL;

  scaled = ARMCII_Buf_acc_is_scaled(datatype, scale);

  if (scaled) {

    char *contig;
    MPI_Alloc_mem((MPI_Aint)count*size, MPI_INFO_NULL, &contig);
    ARMCII_Assert(contig != NULL);
    new_bufs[count] = contig;

    for (i = 0; i < count; i++) {
      new_bufs[i] = contig + (MPI_Aint)i*size;
      ARMCII_Buf_acc_scale(orig_bufs[i], new_bufs[i], size, datatype, scale);
    }
  } else {
    for (i = 0; i < count; i++) {
      gmr_t *mreg = NULL;

      if (ARMCII_GLOBAL_STATE.shr_buf_method != ARMCII_SHR_BUF_NOGUARD)
        mreg = gmr_lookup(orig_bufs[i], ARMCI_GROUP_WORLD.rank);

      new_bufs[i] = orig_bufs[i];

      if (mreg != NULL) {

        MPI_Alloc_mem(size, MPI_INFO_NULL, &new_bufs[i]);
        ARMCII_Assert(new_bufs[i] != NULL);

        ARMCI_Copy(orig_bufs[i], new_bufs[i], size);
      }

      if (new_bufs[i] == orig_bufs[i])
        num_moved++;
    }
  }

  *new_bufs_ptr = new_bufs;

  return num_moved;
}

void ARMCII_Buf_finish_acc_vec(void **orig_bufs, void **new_bufs, int count, int size) {
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

int ARMCII_Buf_acc_is_scaled(int datatype, void *scale) {
  switch (datatype) {
    case ARMCI_ACC_INT:
      if (*((int*)scale) == 1)
        return 0;
      break;

    case ARMCI_ACC_LNG:
      if (*((long*)scale) == 1)
        return 0;
      break;

    case ARMCI_ACC_FLT:
      if (fabsf(*((float*)scale)-1.0f) < FLT_EPSILON)
        return 0;
      break;

    case ARMCI_ACC_DBL:
      if (fabs(*((double*)scale)-1.0) < DBL_EPSILON)
        return 0;
      break;

    case ARMCI_ACC_CPL:
      if (fabsf(((float*)scale)[0]-1.0f) < FLT_EPSILON &&
          fabsf(((float*)scale)[1]-0.0f) < FLT_EPSILON)
        return 0;
      break;

    case ARMCI_ACC_DCP:
      if (fabs(((double*)scale)[0]-1.0) < DBL_EPSILON &&
          fabs(((double*)scale)[1]-0.0) < DBL_EPSILON)
        return 0;
      break;

    default:
      ARMCII_Error("unknown data type (%d)", datatype);
  }

  return 1;
}

void ARMCII_Buf_acc_scale(void *buf_in, void *buf_out, int size, int datatype, void *scale) {
  int   j, nelem;
  int   type_size = -1;

  switch (datatype) {
    case ARMCI_ACC_INT:
      MPI_Type_size(MPI_INT, &type_size);
      nelem= size/type_size;

      {
        int *src_i = (int*) buf_in;
        int *scl_i = (int*) buf_out;
        const int s = *((int*) scale);

        for (j = 0; j < nelem; j++)
          scl_i[j] = src_i[j]*s;
      }
      break;

    case ARMCI_ACC_LNG:
      MPI_Type_size(MPI_LONG, &type_size);
      nelem= size/type_size;

      {
        long *src_l = (long*) buf_in;
        long *scl_l = (long*) buf_out;
        const long s = *((long*) scale);

        for (j = 0; j < nelem; j++)
          scl_l[j] = src_l[j]*s;
      }
      break;

    case ARMCI_ACC_FLT:
      MPI_Type_size(MPI_FLOAT, &type_size);
      nelem= size/type_size;

      {
        float *src_f = (float*) buf_in;
        float *scl_f = (float*) buf_out;
        const float s = *((float*) scale);

        for (j = 0; j < nelem; j++)
          scl_f[j] = src_f[j]*s;
      }
      break;

    case ARMCI_ACC_DBL:
      MPI_Type_size(MPI_DOUBLE, &type_size);
      nelem= size/type_size;

      {
        double *src_d = (double*) buf_in;
        double *scl_d = (double*) buf_out;
        const double s = *((double*) scale);

        for (j = 0; j < nelem; j++)
          scl_d[j] = src_d[j]*s;
      }
      break;

    case ARMCI_ACC_CPL:
      MPI_Type_size(MPI_FLOAT, &type_size);
      nelem= size/type_size;

      {
        float *src_fc = (float*) buf_in;
        float *scl_fc = (float*) buf_out;
        const float s_r = ((float*)scale)[0];
        const float s_c = ((float*)scale)[1];

        for (j = 0; j < nelem; j += 2) {

          const float src_fc_j   = src_fc[j];
          const float src_fc_j_1 = src_fc[j+1];

          scl_fc[j]   = src_fc_j*s_r   - src_fc_j_1*s_c;
          scl_fc[j+1] = src_fc_j_1*s_r + src_fc_j*s_c;
        }
      }
      break;

    case ARMCI_ACC_DCP:
      MPI_Type_size(MPI_DOUBLE, &type_size);
      nelem= size/type_size;

      {
        double *src_dc = (double*) buf_in;
        double *scl_dc = (double*) buf_out;
        const double s_r = ((double*)scale)[0];
        const double s_c = ((double*)scale)[1];

        for (j = 0; j < nelem; j += 2) {

          const double src_dc_j   = src_dc[j];
          const double src_dc_j_1 = src_dc[j+1];

          scl_dc[j]   = src_dc_j*s_r   - src_dc_j_1*s_c;
          scl_dc[j+1] = src_dc_j_1*s_r + src_dc_j*s_c;
        }
      }
      break;

    default:
      ARMCII_Error("unknown data type (%d)", datatype);
  }

  ARMCII_Assert_msg(size % type_size == 0,
      "Transfer size is not a multiple of the datatype size");
}

#define MIN(A,B) (((A) < (B)) ? (A) : (B))

static armcix_mutex_hdl_t armci_mutex_hdl = NULL;

enum ARMCII_MPI_Impl_e { ARMCII_MPICH,
                         ARMCII_OPEN_MPI,
                         ARMCII_MVAPICH2,
                         ARMCII_INTEL_MPI,
                         ARMCII_CRAY_MPI,
                         ARMCII_UNKNOWN_MPI };

static void ARMCII_Parse_library_version(char * library_version, enum ARMCII_MPI_Impl_e * impl,
                                         int * major, int * minor, char * patch)
{
    *impl  = ARMCII_UNKNOWN_MPI;
    *major = -1;
    *minor = -1;

    int is_mpich = 0, is_ompi = 0, is_impi = 0;
    {
        char * p = NULL;
        p = strstr(library_version,"MPICH");
        is_mpich = (p != NULL);
        p = strstr(library_version,"Open MPI");
        is_ompi = (p != NULL);
        p = strstr(library_version,"Intel(R) MPI Library");
        is_impi = (p != NULL);
    }

    if (is_mpich) {
      *impl = ARMCII_MPICH;
      int mpich_major = 0;
      int mpich_minor = 0;
      char mpich_patch[4] = {0};
      for (int major = 9; major >= 3; major--) {
        for (int minor = 9; minor >= 0; minor--) {
          char version_string[4] = {0};
          sprintf(version_string,"%d.%d",major,minor);
          char * p = strstr(library_version,version_string);
          if (p != NULL) {
            mpich_major = atoi(p);
            mpich_minor = atoi(p+2);
            strncpy(mpich_patch,p+3,4);
            break;
          }
        }
      }
      *major = mpich_major;
      *minor = mpich_minor;
      strncpy(patch, mpich_patch, sizeof(mpich_patch));
    }

    if (is_ompi) {
      *impl = ARMCII_OPEN_MPI;
      int ompi_major = 0;
      int ompi_minor = 0;
      char ompi_patch[6] = {0};
      for (int major = 9; major >= 2; major--) {
        for (int minor = 9; minor >= 0; minor--) {
          char version_string[4] = {0};
          sprintf(version_string,"%d.%d",major,minor);
          char * p = strstr(library_version,version_string);
          if (p != NULL) {
            ompi_major = atoi(p);
            ompi_minor = atoi(p+2);
            strncpy(ompi_patch,p+3,4);
            for (int c=0; c<sizeof(ompi_patch); c++) {
              if (ompi_patch[c] == ',') {
                ompi_patch[c] = '\0';
                break;
              }
            }
            break;
          }
        }
      }
      *major = ompi_major;
      *minor = ompi_minor;
      strncpy(patch, ompi_patch, sizeof(ompi_patch));
    }

    if (is_impi) {
      *impl = ARMCII_INTEL_MPI;
      int impi_major = 0;
      int impi_minor = 0;
      char impi_patch[6] = {0};
      for (int major = 2030; major >= 2000; major--) {
        for (int minor = 30; minor >= 0; minor--) {
          char version_string[7] = {0};
          sprintf(version_string,"%d.%d",major,minor);
          char * p = strstr(library_version,version_string);
          if (p != NULL) {
            impi_major = atoi(p);
            impi_minor = atoi(p+5);
            break;
          }
        }
      }
      *major = impi_major;
      *minor = impi_minor;
      *patch = '\0';
    }
}

int PARMCI_Init_thread_comm(int armci_requested, MPI_Comm comm) {

  if (ARMCII_GLOBAL_STATE.init_count > 0) {
    ARMCII_GLOBAL_STATE.init_count++;
    return 0;
  }

  {
    int mpi_is_init, mpi_is_fin;
    MPI_Initialized(&mpi_is_init);
    MPI_Finalized(&mpi_is_fin);
    if (!mpi_is_init || mpi_is_fin) {
      ARMCII_Error("MPI must be initialized before calling ARMCI_Init");
    }
  }

  MPI_Comm_dup(comm, &ARMCI_GROUP_WORLD.comm);
  ARMCII_Group_init_from_comm(&ARMCI_GROUP_WORLD);
  ARMCI_GROUP_DEFAULT = ARMCI_GROUP_WORLD;

  ARMCII_GLOBAL_STATE.verbose = ARMCII_Getenv_int("ARMCI_VERBOSE", 0);

  char mpi_library_version[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
  char mpi_library_version_short[32] = {0};
  enum ARMCII_MPI_Impl_e mpi_implementation;
  int mpi_impl_major = 0;
  int mpi_impl_minor = 0;
  char mpi_impl_patch[8] = {0};
  {
    int len;
    MPI_Get_library_version(mpi_library_version, &len);

    strncpy(mpi_library_version_short, mpi_library_version, 31);
    for (int c=0; c<sizeof(mpi_library_version_short); c++) {
      if (mpi_library_version[c] == '\r' || mpi_library_version[c] == '\n') {
        mpi_library_version_short[c] = '\0';
        break;
      }
    }
    ARMCII_Parse_library_version(mpi_library_version_short, &mpi_implementation,
                                 &mpi_impl_major, &mpi_impl_minor, mpi_impl_patch);
  }

  {
    int mpi_thread_level;
    MPI_Query_thread(&mpi_thread_level);

    if (mpi_thread_level<armci_requested) {
      ARMCII_Error("MPI thread level below ARMCI thread level!");
    }

    ARMCII_GLOBAL_STATE.thread_level = armci_requested;

  }

  ARMCII_GLOBAL_STATE.debug_alloc = ARMCII_Getenv_bool("ARMCI_DEBUG_ALLOC", 0);
  {
    int junk;
    junk = ARMCII_Getenv_bool("ARMCI_FLUSH_BARRIERS", -1);
    if (junk != -1) {
      ARMCII_Warning("ARMCI_FLUSH_BARRIERS is deprecated.\n");
    }
  }

  if (ARMCII_Getenv("ARMCI_NONCOLLECTIVE_GROUPS")) {
    ARMCII_GLOBAL_STATE.noncollective_groups = ARMCII_Getenv_bool("ARMCI_NONCOLLECTIVE_GROUPS", 0);
  }
  ARMCII_GLOBAL_STATE.cache_rank_translation = ARMCII_Getenv_bool("ARMCI_CACHE_RANK_TRANSLATION", 1);

#if defined(OPEN_MPI) && defined(OMPI_MAJOR_VERSION) && (OMPI_MAJOR_VERSION < 5)

  ARMCII_GLOBAL_STATE.iov_method = ARMCII_IOV_BATCHED;
  ARMCII_GLOBAL_STATE.strided_method = ARMCII_STRIDED_IOV;
#else

  ARMCII_GLOBAL_STATE.iov_method = ARMCII_IOV_DIRECT;
  ARMCII_GLOBAL_STATE.strided_method = ARMCII_STRIDED_DIRECT;
#endif

  ARMCII_GLOBAL_STATE.iov_checks        = ARMCII_Getenv_bool("ARMCI_IOV_CHECKS", 0);
  ARMCII_GLOBAL_STATE.iov_batched_limit = ARMCII_Getenv_int("ARMCI_IOV_BATCHED_LIMIT", 0);

  if (ARMCII_GLOBAL_STATE.iov_batched_limit < 0) {
    ARMCII_Warning("Ignoring invalid value for ARMCI_IOV_BATCHED_LIMIT (%d)\n",
                   ARMCII_GLOBAL_STATE.iov_batched_limit);
    ARMCII_GLOBAL_STATE.iov_batched_limit = 0;
  }

#if defined(OPEN_MPI)
# if (OMPI_MAJOR_VERSION >= 5)
  const int iov_dtype_chunk_default = 1;
# else
  const int iov_dtype_chunk_default = 0;
# endif
#elif defined(MPICH_VERSION)
  const int iov_dtype_chunk_default = 256;
#else
  const int iov_dtype_chunk_default = 0;
#endif
  ARMCII_GLOBAL_STATE.iov_dtype_chunk      = ARMCII_Getenv_int("ARMCI_IOV_DTYPE_CHUNK", iov_dtype_chunk_default);
  if (ARMCII_GLOBAL_STATE.iov_dtype_chunk < 0) {
    ARMCII_Warning("Ignoring invalid value for ARMCI_IOV_DTYPE_CHUNK (%d)\n", ARMCII_GLOBAL_STATE.iov_dtype_chunk);
    ARMCII_GLOBAL_STATE.iov_dtype_chunk = 0;
  }

  char *var = ARMCII_Getenv("ARMCI_IOV_METHOD");
  if (var != NULL) {
    if (strcmp(var, "AUTO") == 0)
      ARMCII_GLOBAL_STATE.iov_method = ARMCII_IOV_AUTO;
    else if (strcmp(var, "CONSRV") == 0)
      ARMCII_GLOBAL_STATE.iov_method = ARMCII_IOV_CONSRV;
    else if (strcmp(var, "BATCHED") == 0)
      ARMCII_GLOBAL_STATE.iov_method = ARMCII_IOV_BATCHED;
    else if (strcmp(var, "DIRECT") == 0)
      ARMCII_GLOBAL_STATE.iov_method = ARMCII_IOV_DIRECT;
    else if (ARMCI_GROUP_WORLD.rank == 0)
      ARMCII_Warning("Ignoring unknown value for ARMCI_IOV_METHOD (%s)\n", var);
  }

  var = ARMCII_Getenv("ARMCI_STRIDED_METHOD");
  if (var != NULL) {
    if (strcmp(var, "IOV") == 0) {
      ARMCII_GLOBAL_STATE.strided_method = ARMCII_STRIDED_IOV;
    } else if (strcmp(var, "DIRECT") == 0) {
      ARMCII_GLOBAL_STATE.strided_method = ARMCII_STRIDED_DIRECT;
    } else if (ARMCI_GROUP_WORLD.rank == 0) {
      ARMCII_Warning("Ignoring unknown value for ARMCI_STRIDED_METHOD (%s)\n", var);
    }
  }

#ifdef OPEN_MPI
  if (ARMCII_GLOBAL_STATE.iov_method == ARMCII_IOV_DIRECT || ARMCII_GLOBAL_STATE.strided_method == ARMCII_STRIDED_DIRECT) {
    if (ARMCI_GROUP_WORLD.rank == 0) {
      ARMCII_Warning("MPI Datatypes are broken in RMA in many versions of Open-MPI!\n");
#if defined(OMPI_MAJOR_VERSION) && (OMPI_MAJOR_VERSION == 4)
      ARMCII_Warning("Open-MPI 4.0.0 RMA with datatypes is definitely broken."
                     "See https://github.com/open-mpi/ompi/issues/6275 for details.\n");
#endif
    }
  }
#endif

  ARMCII_GLOBAL_STATE.shr_buf_method = ARMCII_SHR_BUF_NOGUARD;

  var = ARMCII_Getenv("ARMCI_SHR_BUF_METHOD");
  if (var != NULL) {
    if (strcmp(var, "COPY") == 0) {
      ARMCII_GLOBAL_STATE.shr_buf_method = ARMCII_SHR_BUF_COPY;
    } else if (strcmp(var, "NOGUARD") == 0) {
      ARMCII_GLOBAL_STATE.shr_buf_method = ARMCII_SHR_BUF_NOGUARD;
    } else if (ARMCI_GROUP_WORLD.rank == 0) {
      ARMCII_Warning("Ignoring unknown value for ARMCI_SHR_BUF_METHOD (%s)\n", var);
    }
  }

  ARMCII_GLOBAL_STATE.use_win_allocate = ARMCII_Getenv_bool("ARMCI_USE_WIN_ALLOCATE", 1);

  ARMCII_GLOBAL_STATE.msg_barrier_syncs = ARMCII_Getenv_bool("ARMCI_MSG_BARRIER_SYNCS", 0);

  ARMCII_GLOBAL_STATE.memory_limit=ARMCII_Getenv_long("ARMCI_SHM_LIMIT", 0);

  ARMCII_GLOBAL_STATE.explicit_nb_progress=ARMCII_Getenv_bool("ARMCI_EXPLICIT_NB_PROGRESS", 1);

  ARMCII_GLOBAL_STATE.use_alloc_shm=ARMCII_Getenv_bool("ARMCI_USE_ALLOC_SHM", 1);

  ARMCII_GLOBAL_STATE.disable_shm_accumulate=ARMCII_Getenv_bool("ARMCI_DISABLE_SHM_ACC", 0);

  ARMCII_GLOBAL_STATE.use_same_op=ARMCII_Getenv_bool("ARMCI_USE_SAME_OP", 0);

  ARMCII_GLOBAL_STATE.rma_atomicity=ARMCII_Getenv_bool("ARMCI_RMA_ATOMICITY", 1);

  ARMCII_Getenv_char(ARMCII_GLOBAL_STATE.rma_ordering, "ARMCI_RMA_ORDERING", "rar,raw,war,waw",
                     sizeof(ARMCII_GLOBAL_STATE.rma_ordering)-1);

  ARMCII_GLOBAL_STATE.end_to_end_flush=ARMCII_Getenv_bool("ARMCI_NO_FLUSH_LOCAL", 0);

  ARMCII_GLOBAL_STATE.rma_nocheck=ARMCII_Getenv_bool("ARMCI_RMA_NOCHECK", 1);

#if defined(OPEN_MPI) && defined(OMPI_MAJOR_VERSION) && (OMPI_MAJOR_VERSION == 4)
  const int use_request_atomics_default = 0;
#else
  const int use_request_atomics_default = 1;
#endif
  ARMCII_GLOBAL_STATE.use_request_atomics=ARMCII_Getenv_bool("ARMCI_USE_REQUEST_ATOMICS", use_request_atomics_default);
#if defined(OPEN_MPI) && defined(OMPI_MAJOR_VERSION) && (OMPI_MAJOR_VERSION == 4)
  if (ARMCII_GLOBAL_STATE.use_request_atomics) {
      ARMCII_Warning("MPI request-based atomics are buggy with Open-MPI 4.x UCX on IB"
		     " (https://github.com/open-mpi/ompi/issues/14173); "
		     "set ARMCI_USE_REQUEST_ATOMICS=0 to disable.\n");
  }
#endif

  ARMCII_GLOBAL_STATE.flush_request_atomics=ARMCII_Getenv_bool("ARMCI_FLUSH_REQUEST_ATOMICS", 0);

  const int use_rma_requests = 1;

  ARMCII_GLOBAL_STATE.init_count++;

  if (ARMCII_GLOBAL_STATE.verbose > 0) {
    if (ARMCI_GROUP_WORLD.rank == 0) {
      int major, minor;

      MPI_Get_version(&major, &minor);

      printf("ARMCI-MPI initialized with %d process%s, MPI v%d.%d\n",
             ARMCI_GROUP_WORLD.size, ARMCI_GROUP_WORLD.size > 1 ? "es":"", major, minor);

      if (ARMCII_GLOBAL_STATE.verbose > 1) {
        printf("=======\n");
        printf("  MPI library version    = %s", mpi_library_version);
        printf("=======\n");
      } else {
        if (mpi_implementation == ARMCII_OPEN_MPI) {
          printf("  Open MPI version       = %d.%d%s\n", mpi_impl_major, mpi_impl_minor, mpi_impl_patch);
        }
        else if (mpi_implementation == ARMCII_MPICH) {
          printf("  MPICH version          = %d.%d%s\n", mpi_impl_major, mpi_impl_minor, mpi_impl_patch);
        }
        else if (mpi_implementation == ARMCII_INTEL_MPI) {
          printf("  Intel MPI version      = %d.%d%s\n", mpi_impl_major, mpi_impl_minor, mpi_impl_patch);
        }
        else if (mpi_implementation == ARMCII_UNKNOWN_MPI) {
          printf("  Unknown MPI version    = %s\n", mpi_library_version);
        }
      }

      printf("  EXPLICIT_NB_PROGRESS   = %s\n", ARMCII_GLOBAL_STATE.explicit_nb_progress ? "ENABLED" : "DISABLED");

      if (ARMCII_GLOBAL_STATE.memory_limit > 0) {
          size_t limit  = ARMCII_GLOBAL_STATE.memory_limit;
          char suffix[4] = {0};
          int    offset = 0;
          if (limit && !(limit & (limit-1))) {
            char * bsuffix[5] = {"", "KiB","MiB","GiB","TiB"};
            for (int i=0; i<4; i++) {
              if (limit % 1024 == 0) {
                offset++;
                limit /= 1024;
              }
            }
            strncpy(suffix, bsuffix[offset], sizeof(suffix));
          } else {
            char * dsuffix[5] = {"","KB","MB","GB","TB"};
            for (int i=0; i<4; i++) {
              if (limit % 1000 == 0) {
                offset++;
                limit /= 1000;
              }
            }
            strncpy(suffix, dsuffix[offset], sizeof(suffix));
          }
          printf("  SHM_LIMIT              = %zu %s\n", limit, suffix);
      } else {
          printf("  SHM_LIMIT              = %s\n", "UNLIMITED");
      }

      if (ARMCII_GLOBAL_STATE.use_win_allocate == 0) {
          printf("  WINDOW type used       = %s\n", "CREATE");
      }
      else if (ARMCII_GLOBAL_STATE.use_win_allocate == 1) {
          printf("  WINDOW type used       = %s\n", "ALLOCATE");
      }
      else {
          ARMCII_Error("You have selected an invalid window type (%d)!\n", ARMCII_GLOBAL_STATE.use_win_allocate);
      }

      printf("  STRIDED_METHOD         = %s\n", ARMCII_Strided_methods_str[ARMCII_GLOBAL_STATE.strided_method]);
      printf("  IOV_METHOD             = %s\n", ARMCII_Iov_methods_str[ARMCII_GLOBAL_STATE.iov_method]);

      if (ARMCII_GLOBAL_STATE.iov_method == ARMCII_IOV_BATCHED || ARMCII_GLOBAL_STATE.iov_method == ARMCII_IOV_AUTO) {
        if (ARMCII_GLOBAL_STATE.iov_batched_limit > 0) {
          printf("  IOV_BATCHED_LIMIT      = %d\n", ARMCII_GLOBAL_STATE.iov_batched_limit);
        } else {
          printf("  IOV_BATCHED_LIMIT      = UNLIMITED\n");
        }
      }

      if (ARMCII_GLOBAL_STATE.iov_method == ARMCII_IOV_DIRECT || ARMCII_GLOBAL_STATE.iov_method == ARMCII_IOV_AUTO) {
        if (ARMCII_GLOBAL_STATE.iov_dtype_chunk > 0)
          printf("  IOV_DTYPE_CHUNK        = %d\n", ARMCII_GLOBAL_STATE.iov_dtype_chunk);
        else
          printf("  IOV_DTYPE_CHUNK        = UNLIMITED\n");
      }

      printf("  RMA_ATOMICITY          = %s\n", ARMCII_GLOBAL_STATE.rma_atomicity          ? "TRUE" : "FALSE");
      printf("  NO_FLUSH_LOCAL         = %s\n", ARMCII_GLOBAL_STATE.end_to_end_flush       ? "TRUE" : "FALSE");
      printf("  RMA_NOCHECK            = %s\n", ARMCII_GLOBAL_STATE.rma_nocheck            ? "TRUE" : "FALSE");
      printf("  MSG_BARRIER_SYNCS      = %s\n", ARMCII_GLOBAL_STATE.msg_barrier_syncs      ? "TRUE" : "FALSE");
      printf("  USE_REQUEST_ATOMICS    = %s\n", ARMCII_GLOBAL_STATE.use_request_atomics    ? "TRUE" : "FALSE");
      printf("  FLUSH_REQUEST_ATOMICS  = %s\n", ARMCII_GLOBAL_STATE.flush_request_atomics  ? "TRUE" : "FALSE");
      printf("  USE_RMA_REQUESTS       = %s\n", use_rma_requests ? "TRUE" : "FALSE");

      printf("  USE_ALLOC_SHM          = %s\n", ARMCII_GLOBAL_STATE.use_alloc_shm          ? "TRUE" : "FALSE");
      printf("  DISABLE_SHM_ACC        = %s\n", ARMCII_GLOBAL_STATE.disable_shm_accumulate ? "TRUE" : "FALSE");
      printf("  USE_SAME_OP            = %s\n", ARMCII_GLOBAL_STATE.use_same_op            ? "TRUE" : "FALSE");
      printf("  RMA_ORDERING           = %s\n", ARMCII_GLOBAL_STATE.rma_ordering);

      printf("  IOV_CHECKS             = %s\n", ARMCII_GLOBAL_STATE.iov_checks             ? "TRUE" : "FALSE");
      printf("  SHR_BUF_METHOD         = %s\n", ARMCII_Shr_buf_methods_str[ARMCII_GLOBAL_STATE.shr_buf_method]);
      printf("  NONCOLLECTIVE_GROUPS   = %s\n", ARMCII_GLOBAL_STATE.noncollective_groups   ? "TRUE" : "FALSE");
      printf("  CACHE_RANK_TRANSLATION = %s\n", ARMCII_GLOBAL_STATE.cache_rank_translation ? "TRUE" : "FALSE");
      printf("  DEBUG_ALLOC            = %s\n", ARMCII_GLOBAL_STATE.debug_alloc            ? "TRUE" : "FALSE");
      printf("\n");
      fflush(NULL);
    }

    if ((ARMCII_GLOBAL_STATE.use_win_allocate == 1) && (ARMCI_GROUP_WORLD.rank == 0)) {

        printf("  Warning: MPI_Win_allocate can lead to correctness issues.\n");
        if ((mpi_implementation == ARMCII_MPICH) && (mpi_impl_major == 4)) {
          printf("           There is a good chance your implementation is affected!\n");
          printf("           See https://github.com/pmodels/mpich/issues/6110 for details.\n");
        }
        printf("\n");
        fflush(NULL);
    }

    MPI_Barrier(ARMCI_GROUP_WORLD.comm);
  }

  return 0;
}

int PARMCI_Init(void) {
  return PARMCI_Init_thread_comm(MPI_THREAD_SINGLE, MPI_COMM_WORLD);
}

int PARMCI_Initialized(void) {
  return ARMCII_GLOBAL_STATE.init_count > 0;
}

int PARMCI_Finalize(void) {
  int nfreed;

  if (ARMCII_GLOBAL_STATE.init_count == 0) {
    return 0;
  }

  ARMCII_GLOBAL_STATE.init_count--;

  if (ARMCII_GLOBAL_STATE.init_count > 0) {
    return 0;
  }

  nfreed = gmr_destroy_all();

  if (nfreed > 0 && ARMCI_GROUP_WORLD.rank == 0) {
    ARMCII_Warning("Freed %d leaked allocations\n", nfreed);
  }

  ARMCI_Cleanup();

  ARMCI_Group_free(&ARMCI_GROUP_WORLD);

  return 0;
}

void ARMCI_Cleanup(void) {
  return;
}

#define MP_BARRIER()      MPI_Barrier(MPI_COMM_WORLD)
#define ELEMS 500
#define MAXPROC 128
#define TIMES 100
#define BASE 100.
#define ABS(a) (((a) <0) ? -(a) : (a))
#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define MAXDIMS 7
#define OFF 1
#define DIM1 5
#define DIM2 3
#define DIM3 8
#define DIM4 9
#define DIM5 7
#define DIM6 3
#define DIM7 2
#define EDIM1 (DIM1+OFF)
#define EDIM2 (DIM2+OFF)
#define EDIM3 (DIM3+OFF)
#define EDIM4 (DIM4+OFF)
#define EDIM5 (DIM5+OFF)
#define EDIM6 (DIM6+OFF)
#define EDIM7 (DIM7+OFF)
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
		assert(diff == (hi2[i]-lo2[i]));
		assert(diff < dims1[i]);
		assert(diff < dims2[i]);
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
		assert(diff < dims1[i]);
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

     assert(ndim<=MAXDIMS);
     for(i=0;i<ndim;i++)bytes*=dims[i];
     rc = PARMCI_Malloc(a, bytes);
     assert(rc==0);
     assert(a[me]);

}

void destroy_array(void *ptr[])
{
    int rc;
    MP_BARRIER();
    rc = PARMCI_Free(ptr[me]);
    assert(rc==0);
}

int loA[MAXDIMS], hiA[MAXDIMS];
int dimsA[MAXDIMS]={DIM1,DIM2,DIM3,DIM4,DIM5,DIM6, DIM7};
int loB[MAXDIMS], hiB[MAXDIMS];
int dimsB[MAXDIMS]={EDIM1,EDIM2,EDIM3,EDIM4,EDIM5,EDIM6,EDIM7};
int count[MAXDIMS];
int strideA[MAXDIMS], strideB[MAXDIMS];
int loC[MAXDIMS], hiC[MAXDIMS];
int idx[MAXDIMS]={0,0,0,0,0,0,0};

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
        assert(a);
        c = malloc(bytes);
        assert(c);

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

        MP_BARRIER();
        for(i=0;i<TIMES*nproc;i++){

            proc=0;

            for(j=0; j<elems/2; j++){
                psrc[j]= 2*j + (double*)a;
                pdst[j]= 2*j + (double*)b[proc];
            }
            if((rc = PARMCI_AccV(ARMCI_ACC_DBL, &alpha, &dsc, 1, proc)))
                     ARMCI_Error("accumlate failed",rc);

            for(j=0; j< elems/2; j++){
                psrc[j]= 2*j+1 + (double*)a;
                pdst[j]= 2*j+1 + (double*)b[proc];
            }
            (void)PARMCI_AccV(ARMCI_ACC_DBL, &alpha, &dsc, 1, proc);

        }

        PARMCI_AllFence();
        MP_BARRIER();

	assert(!PARMCI_Get((double*)b[proc], c, bytes, proc));

        scale = alpha*TIMES*nproc*nproc;
        scale_patch(scale, dim, a, &one, &elems, &elems);

        compare_patches(.0001, dim, a, &one, &elems, &elems, c, &one, &elems, &elems);
        MP_BARRIER();

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
    PARMCI_Init();
    if (me == 0) printf("standalone test_vector_acc: %d procs\n", nproc);
    test_vector_acc();
    MPI_Barrier(MPI_COMM_WORLD);
    if (me == 0) printf("DONE OK\n");
    PARMCI_Finalize();
    MPI_Finalize();
    return 0;
}