#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <numeric>
#include <time.h>

#define NRUNS 10

#include <hip/hip_runtime.h>
#include <rocsparse.h>
#include <rocsparse-complex-types.h>

#undef NDEBUG

using complex_t = std::complex<double>;
const complex_t h_one = 1.0;
const rocsparse_analysis_policy an_policy = rocsparse_analysis_policy_reuse;

#define CHECK(cmd)                                                            \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

struct problem
{
    int nrows;
    int nnz;
    int nrhs;
    int *row_ptr;
    int *col_ind;
    complex_t *val;
    complex_t *rhs;
    complex_t *sol;
    complex_t *sol_m;
};

struct problem read_problem()
{
  struct problem p;

  std::ifstream f("sparse_matrix.dat", std::fstream::in);

  size_t nbytes;
  int device_id;
  CHECK(hipGetDevice(&device_id));

  f >> p.nrows;
  std::cout << "nrows " << p.nrows << std::endl;
  nbytes = (p.nrows + 1) * sizeof(int);
  CHECK(hipMallocManaged(&p.row_ptr, nbytes));
  CHECK(hipMemAdvise(p.row_ptr, nbytes, hipMemAdviseSetCoarseGrain, device_id));
  for (int i = 0; i < p.nrows + 1; i++) {
    f >> p.row_ptr[i];
  }
  f >> p.nnz;
  std::cout << "nnz " << p.nnz << std::endl;
  nbytes = p.nnz * sizeof(int);
  CHECK(hipMallocManaged(&p.col_ind, nbytes));
  CHECK(hipMemAdvise(p.col_ind, nbytes, hipMemAdviseSetCoarseGrain, device_id));
  nbytes = p.nnz * sizeof(complex_t);
  CHECK(hipMallocManaged(&p.val, nbytes));
  CHECK(hipMemAdvise(p.val, nbytes, hipMemAdviseSetCoarseGrain, device_id));
  for (int i = 0; i < p.nnz; i++) {
    f >> p.col_ind[i];
  }
  for (int i = 0; i < p.nnz; i++) {
    f >> p.val[i];
  }
  f >> p.nrhs;
  std::cout << "nrhs " << p.nrhs << std::endl;
  nbytes = p.nnz * p.nrhs * sizeof(complex_t);
  CHECK(hipMallocManaged(&p.rhs, nbytes));
  CHECK(hipMemAdvise(p.rhs, nbytes, hipMemAdviseSetCoarseGrain, device_id));
  CHECK(hipMallocManaged(&p.sol, nbytes));
  CHECK(hipMemAdvise(p.sol, nbytes, hipMemAdviseSetCoarseGrain, device_id));
  for (int i = 0; i < p.nnz * p.nrhs; i++) {
    f >> p.rhs[i];
  }

  return p;
}

struct problem problem_to_device(struct problem p)
{
  struct problem d_p = p;
  int device_id;

  CHECK(hipGetDevice(&device_id));

  CHECK(hipMalloc(&d_p.row_ptr, (p.nrows + 1) * sizeof(int)));
  CHECK(hipMalloc(&d_p.col_ind, p.nnz * sizeof(int)));
  CHECK(hipMalloc(&d_p.val, p.nnz * sizeof(complex_t)));
  // rhs is always managed
  // CHECK(hipMalloc(&d_p.rhs, p.nnz * p.nrhs * sizeof(complex_t)));
  CHECK(hipMalloc(&d_p.sol, p.nnz * p.nrhs * sizeof(complex_t)));
  CHECK(hipMallocManaged(&d_p.sol_m, p.nnz * p.nrhs * sizeof(complex_t)));
  CHECK(hipMemAdvise(d_p.sol_m, p.nnz * p.nrhs * sizeof(complex_t),
                     hipMemAdviseSetCoarseGrain, device_id));

  CHECK(hipMemcpy(d_p.row_ptr, p.row_ptr, (p.nrows + 1) * sizeof(int), hipMemcpyDeviceToDevice));
  CHECK(hipMemcpy(d_p.col_ind, p.col_ind, p.nnz * sizeof(int), hipMemcpyDeviceToDevice));
  CHECK(hipMemcpy(d_p.val, p.val, p.nnz * sizeof(complex_t), hipMemcpyDeviceToDevice));
  CHECK(hipMemcpy(d_p.rhs, p.rhs, p.nnz * p.nrhs * sizeof(complex_t), hipMemcpyDeviceToDevice));

  hipDeviceSynchronize();

  return d_p;
}

void solve(rocsparse_handle h, struct problem p, bool copy_out=false)
{
  rocsparse_status status;
  rocsparse_mat_descr l_mat_descr;
  rocsparse_mat_info l_mat_info;
  rocsparse_mat_descr u_mat_descr;
  rocsparse_mat_info u_mat_info;
  size_t l_work_bytes, u_work_bytes;
  char *l_work, *u_work;

  status = rocsparse_create_mat_info(&l_mat_info);
  assert(rocsparse_status_success == status);
  status = rocsparse_create_mat_info(&u_mat_info);
  assert(rocsparse_status_success == status);

  /* setup lower matrix of LU */
  status = rocsparse_create_mat_descr(&l_mat_descr);
  assert(rocsparse_status_success == status);

  rocsparse_set_mat_index_base(l_mat_descr, rocsparse_index_base_one);
  rocsparse_set_mat_type(l_mat_descr, rocsparse_matrix_type_general);
  rocsparse_set_mat_fill_mode(l_mat_descr, rocsparse_fill_mode_lower);
  rocsparse_set_mat_diag_type(l_mat_descr, rocsparse_diag_type_unit);

  status = rocsparse_zcsrsm_buffer_size(h,
    rocsparse_operation_none,
    rocsparse_operation_none,
    p.nrows, p.nrhs, p.nnz, (rocsparse_double_complex*)&h_one, l_mat_descr,
    (rocsparse_double_complex*)p.val, p.row_ptr, p.col_ind,
    (rocsparse_double_complex*)p.rhs, p.nrows,
    l_mat_info, rocsparse_solve_policy_auto, &l_work_bytes);
  assert(rocsparse_status_success == status);

  hipError_t error = hipMalloc((void**)&(l_work), l_work_bytes);
  assert(hipSuccess == error);

  status = rocsparse_zcsrsm_analysis(h,
    rocsparse_operation_none,
    rocsparse_operation_none,
    p.nrows, p.nrhs, p.nnz, (rocsparse_double_complex*)&h_one, l_mat_descr,
    (rocsparse_double_complex*)p.val, p.row_ptr, p.col_ind,
    (rocsparse_double_complex*)p.rhs, p.nrows, /* ldb */
    l_mat_info, an_policy, rocsparse_solve_policy_auto,
    l_work);
  assert(rocsparse_status_success == status);

  /* setup upper matrix of LU */
  status = rocsparse_create_mat_descr(&u_mat_descr);
  assert(rocsparse_status_success == status);
  rocsparse_set_mat_index_base(u_mat_descr, rocsparse_index_base_one);
  rocsparse_set_mat_type(u_mat_descr, rocsparse_matrix_type_general);
  rocsparse_set_mat_fill_mode(u_mat_descr, rocsparse_fill_mode_upper);
  rocsparse_set_mat_diag_type(u_mat_descr, rocsparse_diag_type_non_unit);

  status = rocsparse_zcsrsm_buffer_size(h,
    rocsparse_operation_none,
    rocsparse_operation_none,
    p.nrows, p.nrhs, p.nnz, (rocsparse_double_complex*)&h_one, u_mat_descr,
    (rocsparse_double_complex*)p.val, p.row_ptr, p.col_ind,
    (rocsparse_double_complex*)p.rhs, p.nrows,
    u_mat_info, rocsparse_solve_policy_auto, &(u_work_bytes));
  assert(rocsparse_status_success == status);

  error = hipMalloc((void**)&(u_work), u_work_bytes);
  assert(hipSuccess == error);

  status = rocsparse_zcsrsm_analysis(h,
    rocsparse_operation_none,
    rocsparse_operation_none,
    p.nrows, p.nrhs, p.nnz, (rocsparse_double_complex*)&h_one, u_mat_descr,
    (rocsparse_double_complex*)p.val, p.row_ptr, p.col_ind,
    (rocsparse_double_complex*)p.rhs, p.nrows, /* ldb */
    u_mat_info, an_policy, rocsparse_solve_policy_auto,
    u_work);
  assert(rocsparse_status_success == status);

  struct timespec start, end;
  double elapsed;

  /* step 1: solve L * Y = B */
  clock_gettime(CLOCK_MONOTONIC, &start);

  hipMemcpy(p.sol, p.rhs, p.nrows * p.nrhs * sizeof(complex_t),
            hipMemcpyDeviceToDevice);
  status = rocsparse_zcsrsm_solve(h,
    rocsparse_operation_none,
    rocsparse_operation_none,
    p.nrows, p.nrhs, p.nnz, (rocsparse_double_complex*)&h_one, l_mat_descr,
    (rocsparse_double_complex*)p.val, p.row_ptr, p.col_ind,
    (rocsparse_double_complex*)p.sol, p.nrows,
    l_mat_info, rocsparse_solve_policy_auto,
    l_work);
  assert(rocsparse_status_success == status);

  /* step 2: solve U * X = Y */
  status = rocsparse_zcsrsm_solve(h,
    rocsparse_operation_none,
    rocsparse_operation_none,
    p.nrows, p.nrhs, p.nnz, (rocsparse_double_complex*)&h_one, u_mat_descr,
    (rocsparse_double_complex*)p.val, p.row_ptr, p.col_ind,
    (rocsparse_double_complex*)p.sol, p.nrows,
    u_mat_info, rocsparse_solve_policy_auto,
    u_work);
  assert(rocsparse_status_success == status);

  if (copy_out) {
    hipMemcpy(p.sol_m, p.sol, p.nnz * p.nrhs * sizeof(complex_t),
              hipMemcpyDeviceToDevice);
  }

  CHECK(hipDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  // count++;
  elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
  // total += elapsed;
  // std::cout << "sparse " << count << " " << elapsed << " " << total / count << std::endl;
  std::cout << "sparse " << elapsed << std::endl;
}

int main (int argc, char *argv[])
{
  struct problem p;
  rocsparse_handle h = 0;
  int error;

  error = (int)rocsparse_create_handle(&h);
  assert(rocsparse_status_success == error);

  p = read_problem();

  std::cout << "nrows " << p.nrows << std::endl;
  std::cout << "nnz " << p.nnz << std::endl;
  std::cout << "nrhs " << p.nrhs << std::endl;

  std::cout << "managed warmup run" << std::endl;
  solve(h, p);

  std::cout << "managed memory runs" << std::endl;
  solve(h, p);
  // std::cout << p.rhs[0] << std::endl;
  solve(h, p);
  // std::cout << p.rhs[0] << std::endl;
  solve(h, p);
  // std::cout << p.rhs[0] << std::endl;
  solve(h, p);

  auto d_p = problem_to_device(p);

  std::cout << "device warmup run" << std::endl;
  solve(h, d_p, true);
  std::cout << "device memory runs" << std::endl;
  solve(h, d_p, true);
  solve(h, d_p, true);
  solve(h, d_p, true);
  solve(h, d_p, true);

  return 0;
}
