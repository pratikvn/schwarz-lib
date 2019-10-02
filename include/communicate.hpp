#ifndef communicate_hpp
#define communicate_hpp

#include "initialization.hpp"

namespace SchwarzWrappers {
template <typename ValueType, typename IndexType> class Communicate {
public:
  // Communicate();

  // virtual ~Communicate() = default;

  friend class Initialize<ValueType, IndexType>;

  struct comm_struct {
  public:
    int num_neighbors_in;
    int num_neighbors_out;
    std::shared_ptr<gko::Array<IndexType>> neighbors_in;
    std::shared_ptr<gko::Array<IndexType>> neighbors_out;

    std::shared_ptr<gko::Array<IndexType *>> global_put;
    std::shared_ptr<gko::Array<IndexType *>> local_put;
    std::shared_ptr<gko::Array<IndexType *>> remote_put;

    std::shared_ptr<gko::Array<IndexType *>> global_get;
    std::shared_ptr<gko::Array<IndexType *>> local_get;
    std::shared_ptr<gko::Array<IndexType *>> remote_get;

    std::shared_ptr<gko::Array<IndexType>> window_ids;
    std::shared_ptr<gko::Array<IndexType>> windows_from;
    std::shared_ptr<gko::Array<IndexType>> windows_to;

    std::shared_ptr<gko::Array<MPI_Request>> put_request;
    std::shared_ptr<gko::Array<MPI_Request>> get_request;

    std::shared_ptr<gko::matrix::Dense<ValueType>> send_buffer;
    std::shared_ptr<gko::matrix::Dense<ValueType>> recv_buffer;

    std::shared_ptr<gko::Array<IndexType>> get_displacements;
    std::shared_ptr<gko::Array<IndexType>> put_displacements;

    MPI_Win window_buffer;
    MPI_Win window_x;

    // comm_struct() = default();
  };

  comm_struct comm_struct;

  virtual void setup_comm_buffers() = 0;

  virtual void setup_windows(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &main_buffer) = 0;

  virtual void exchange_boundary(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector) = 0;

  void local_to_global_vector(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_vector,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &global_vector);

  virtual void update_boundary(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
      const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs,
      const std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &global_old_solution,
      const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
          &interface_matrix) = 0;

  void clear(Settings &settings);
};

} // namespace SchwarzWrappers

#endif
/*----------------------------   communicate.hpp ---------------------------*/
