// This is a modified version of the topology example given in
// the hwloc documentation
// https://www.open-mpi.org/projects/hwloc/doc/v2.0.4/a00312.php#interface

#ifndef process_topology_hpp
#define process_topology_hpp

#include <errno.h>
#include <hwloc.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <schwarz/config.hpp>

#if SCHWARZ_BUILD_CUDA
#include <cuda_runtime.h>
#endif

namespace ProcessTopology {
static void print_children(hwloc_topology_t topology, hwloc_obj_t obj,
                           int depth) {
  char type[32], attr[1024];
  unsigned i;
  hwloc_obj_type_snprintf(type, sizeof(type), obj, 0);
  printf("%*s%s", 2 * depth, "", type);
  if (obj->os_index != (unsigned)-1)
    printf("#%u", obj->os_index);
  hwloc_obj_attr_snprintf(attr, sizeof(attr), obj, " ", 0);
  if (*attr)
    printf("(%s)", attr);
  printf("\n");
  for (i = 0; i < obj->arity; i++) {
    print_children(topology, obj->children[i], depth + 1);
  }
}
void bind_threads_to_process(int local_rank, int local_num_procs,
                             int num_threads) {
  int depth;
  unsigned i, n;
  unsigned long size;
  int levels;
  char string[128];
  int topodepth;
  void *m;
  hwloc_topology_t topology;
  hwloc_cpuset_t cpuset;

  /* Allocate and initialize topology object. */
  hwloc_topology_init(&topology);
  /* Perform the topology detection. */
  hwloc_topology_load(topology);
  /* Optionally, get some additional topology information
     in case we need the topology depth later. */
  topodepth = hwloc_topology_get_depth(topology);
  hwloc_obj_t obj;
  // int         local_rank, local_num_procs;
  // MPI_Comm_rank(local_comm, &local_rank);
  // MPI_Comm_size(local_comm, &local_num_procs);
  int socket_depth = hwloc_get_type_or_below_depth(topology, HWLOC_OBJ_PACKAGE);
  int num_sockets = hwloc_get_nbobjs_by_depth(topology, socket_depth);
  int num_procs_per_socket = local_num_procs;
  depth = hwloc_get_type_or_below_depth(topology, HWLOC_OBJ_CORE);
  omp_set_num_threads(num_threads);
#pragma omp parallel
  {
    /* Get each core. */
    obj = hwloc_get_obj_by_depth(
        topology, depth, omp_get_thread_num() + local_rank * num_threads);
    if (obj) {
      /* Get a copy of its cpuset that we may modify. */
      cpuset = hwloc_bitmap_dup(obj->cpuset);
      /* Get only one logical processor (in case the core is
         SMT/hyper-threaded). */
      hwloc_bitmap_singlify(cpuset);
      /* And try to bind ourself there. */
      if (hwloc_set_cpubind(topology, cpuset, 0)) {
        char *str;
        int error = errno;
        hwloc_bitmap_asprintf(&str, obj->cpuset);
        printf("Couldn't bind to cpuset %s: %s\n", str, strerror(error));
        free(str);
      }
      /* Free our cpuset copy */
      hwloc_bitmap_free(cpuset);
    }
  }
  /* Destroy topology object. */
  hwloc_topology_destroy(topology);
}

void bind_gpus_to_process(int &local_rank, int &local_num_procs,
                          int &num_threads) {
#if SCHWARZ_BUILD_CUDA
  SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(local_rank));
  SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaGetLastError());
#endif
}
} // namespace ProcessTopology

// #define INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro) \
//   template _macro(float, gko::int32);                     \
//   template _macro(double, gko::int32);                    \
//   template _macro(float, gko::int64);                     \
//   template _macro(double, gko::int64);

// #define DECLARE_PROCESSTOPOLOGY(ValueType, IndexType) \
//   struct ProcessTopology<ValueType, IndexType>
// INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_PROCESSTOPOLOGY);
// #undef DECLARE_PROCESSTOPOLOGY

#endif
