
/*******************************<SCHWARZ LIB LICENSE>***********************
Copyright (c) 2019, the SCHWARZ LIB authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<SCHWARZ LIB LICENSE>*************************/


#include <schwarz/config.hpp>


#include <comm_helpers.hpp>
#include <exception_helpers.hpp>
#include <process_topology.hpp>
#include <restricted_schwarz.hpp>
#include <utils.hpp>

#define CHECK_HERE std::cout << "Here " << __LINE__ << std::endl;

namespace schwz {

template <typename ValueType, typename IndexType>
SolverRAS<ValueType, IndexType>::SolverRAS(
    Settings &settings, Metadata<ValueType, IndexType> &metadata)
    : SchwarzBase<ValueType, IndexType>(settings, metadata)
{}


template <typename ValueType, typename IndexType>
void SolverRAS<ValueType, IndexType>::setup_local_matrices(
    Settings &settings, Metadata<ValueType, IndexType> &metadata,
    std::vector<unsigned int> &partition_indices,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &interface_matrix)
{
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using vec_itype = gko::Array<IndexType>;
    using perm_type = gko::matrix::Permutation<IndexType>;
    using arr = gko::Array<IndexType>;
    auto my_rank = metadata.my_rank;
    auto comm_size = metadata.comm_size;
    auto num_subdomains = metadata.num_subdomains;
    auto global_size = metadata.global_size;
    auto mpi_itype = boost::mpi::get_mpi_datatype(*partition_indices.data());

    MPI_Bcast(partition_indices.data(), global_size, mpi_itype, 0,
              MPI_COMM_WORLD);

    std::vector<IndexType> local_p_size(num_subdomains);
    auto global_to_local = metadata.global_to_local->get_data();
    auto local_to_global = metadata.local_to_global->get_data();

    auto first_row = metadata.first_row->get_data();
    auto permutation = metadata.permutation->get_data();
    auto i_permutation = metadata.i_permutation->get_data();

    auto nb = (global_size + num_subdomains - 1) / num_subdomains;
    auto partition_settings =
        (Settings::partition_settings::partition_zoltan |
         Settings::partition_settings::partition_metis |
         Settings::partition_settings::partition_regular |
         Settings::partition_settings::partition_regular2d |
         Settings::partition_settings::partition_custom) &
        settings.partition;

    IndexType *gmat_row_ptrs = global_matrix->get_row_ptrs();
    IndexType *gmat_col_idxs = global_matrix->get_col_idxs();
    ValueType *gmat_values = global_matrix->get_values();

    // default local p size set for 1 subdomain.
    first_row[0] = 0;
    for (auto p = 0; p < num_subdomains; ++p) {
        local_p_size[p] = std::min(global_size - first_row[p], nb);
        first_row[p + 1] = first_row[p] + local_p_size[p];
    }


    if (partition_settings == Settings::partition_settings::partition_metis ||
        partition_settings ==
            Settings::partition_settings::partition_regular2d) {
        if (num_subdomains > 1) {
            for (auto p = 0; p < num_subdomains; p++) {
                local_p_size[p] = 0;
            }
            for (auto i = 0; i < global_size; i++) {
                local_p_size[partition_indices[i]]++;
            }
            first_row[0] = 0;
            for (auto p = 0; p < num_subdomains; ++p) {
                first_row[p + 1] = first_row[p] + local_p_size[p];
            }
            // permutation
            for (auto i = 0; i < global_size; i++) {
                permutation[first_row[partition_indices[i]]] = i;
                first_row[partition_indices[i]]++;
            }
            for (auto p = num_subdomains; p > 0; p--) {
                first_row[p] = first_row[p - 1];
            }
            first_row[0] = 0;

            // iperm
            for (auto i = 0; i < global_size; i++) {
                i_permutation[permutation[i]] = i;
            }
        }

        auto gmat_temp = mtx::create(settings.executor->get_master(),
                                     global_matrix->get_size(),
                                     global_matrix->get_num_stored_elements());

        auto nnz = 0;
        gmat_temp->get_row_ptrs()[0] = 0;
        for (auto row = 0; row < metadata.global_size; ++row) {
            for (auto col = gmat_row_ptrs[permutation[row]];
                 col < gmat_row_ptrs[permutation[row] + 1]; ++col) {
                gmat_temp->get_col_idxs()[nnz] =
                    i_permutation[gmat_col_idxs[col]];
                gmat_temp->get_values()[nnz] = gmat_values[col];
                nnz++;
            }
            gmat_temp->get_row_ptrs()[row + 1] = nnz;
        }
        global_matrix->copy_from(gmat_temp.get());
    }


    for (auto i = 0; i < global_size; i++) {
        global_to_local[i] = 0;
        local_to_global[i] = 0;
    }
    auto num = 0;
    for (auto i = first_row[my_rank]; i < first_row[my_rank + 1]; i++) {
        global_to_local[i] = 1 + num;
        local_to_global[num] = i;
        num++;
    }

    IndexType old = 0;
    for (auto k = 1; k < settings.overlap; k++) {
        auto now = num;
        for (auto i = old; i < now; i++) {
            for (auto j = gmat_row_ptrs[local_to_global[i]];
                 j < gmat_row_ptrs[local_to_global[i] + 1]; j++) {
                if (global_to_local[gmat_col_idxs[j]] == 0) {
                    local_to_global[num] = gmat_col_idxs[j];
                    global_to_local[gmat_col_idxs[j]] = 1 + num;
                    num++;
                }
            }
        }
        old = now;
    }
    metadata.local_size = local_p_size[my_rank];
    metadata.local_size_x = num;
    metadata.local_size_o = global_size;
    auto local_size = metadata.local_size;
    auto local_size_x = metadata.local_size_x;

    metadata.overlap_size = num - metadata.local_size;
    metadata.overlap_row = std::shared_ptr<vec_itype>(
        new vec_itype(gko::Array<IndexType>::view(
            settings.executor, metadata.overlap_size,
            &(metadata.local_to_global->get_data()[metadata.local_size]))),
        std::default_delete<vec_itype>());

    auto nnz_local = 0;
    auto nnz_interface = 0;

    for (auto i = first_row[my_rank]; i < first_row[my_rank + 1]; ++i) {
        for (auto j = gmat_row_ptrs[i]; j < gmat_row_ptrs[i + 1]; j++) {
            if (global_to_local[gmat_col_idxs[j]] != 0) {
                nnz_local++;
            } else {
                std::cout << " debug: invalid edge?" << std::endl;
            }
        }
    }
    auto temp = 0;
    for (auto k = 0; k < metadata.overlap_size; k++) {
        temp = metadata.overlap_row->get_data()[k];
        for (auto j = gmat_row_ptrs[temp]; j < gmat_row_ptrs[temp + 1]; j++) {
            if (global_to_local[gmat_col_idxs[j]] != 0) {
                nnz_local++;
            } else {
                nnz_interface++;
            }
        }
    }

    std::shared_ptr<mtx> local_matrix_compute;
    local_matrix_compute = mtx::create(settings.executor->get_master(),
                                       gko::dim<2>(local_size_x), nnz_local);
    IndexType *lmat_row_ptrs = local_matrix_compute->get_row_ptrs();
    IndexType *lmat_col_idxs = local_matrix_compute->get_col_idxs();
    ValueType *lmat_values = local_matrix_compute->get_values();

    std::shared_ptr<mtx> interface_matrix_compute;
    if (nnz_interface > 0) {
        interface_matrix_compute =
            mtx::create(settings.executor->get_master(),
                        gko::dim<2>(local_size_x), nnz_interface);
    } else {
        interface_matrix_compute = mtx::create(settings.executor->get_master());
    }

    IndexType *imat_row_ptrs = interface_matrix_compute->get_row_ptrs();
    IndexType *imat_col_idxs = interface_matrix_compute->get_col_idxs();
    ValueType *imat_values = interface_matrix_compute->get_values();

    num = 0;
    nnz_local = 0;
    auto nnz_interface_temp = 0;
    lmat_row_ptrs[0] = nnz_local;
    if (nnz_interface > 0) {
        imat_row_ptrs[0] = nnz_interface_temp;
    }
    // Local interior matrix
    for (auto i = first_row[my_rank]; i < first_row[my_rank + 1]; ++i) {
        for (auto j = gmat_row_ptrs[i]; j < gmat_row_ptrs[i + 1]; ++j) {
            if (global_to_local[gmat_col_idxs[j]] != 0) {
                lmat_col_idxs[nnz_local] =
                    global_to_local[gmat_col_idxs[j]] - 1;
                lmat_values[nnz_local] = gmat_values[j];
                nnz_local++;
            }
        }
        if (nnz_interface > 0) {
            imat_row_ptrs[num + 1] = nnz_interface_temp;
        }
        lmat_row_ptrs[num + 1] = nnz_local;
        num++;
    }

    // Interface matrix
    if (nnz_interface > 0) {
        nnz_interface = 0;
        for (auto k = 0; k < metadata.overlap_size; k++) {
            temp = metadata.overlap_row->get_data()[k];
            for (auto j = gmat_row_ptrs[temp]; j < gmat_row_ptrs[temp + 1];
                 j++) {
                if (global_to_local[gmat_col_idxs[j]] != 0) {
                    lmat_col_idxs[nnz_local] =
                        global_to_local[gmat_col_idxs[j]] - 1;
                    lmat_values[nnz_local] = gmat_values[j];
                    nnz_local++;
                } else {
                    imat_col_idxs[nnz_interface] = gmat_col_idxs[j];
                    imat_values[nnz_interface] = gmat_values[j];
                    nnz_interface++;
                }
            }
            lmat_row_ptrs[num + 1] = nnz_local;
            imat_row_ptrs[num + 1] = nnz_interface;
            num++;
        }
    }
    auto now = num;
    for (auto i = old; i < now; i++) {
        for (auto j = gmat_row_ptrs[local_to_global[i]];
             j < gmat_row_ptrs[local_to_global[i] + 1]; j++) {
            if (global_to_local[gmat_col_idxs[j]] == 0) {
                local_to_global[num] = gmat_col_idxs[j];
                global_to_local[gmat_col_idxs[j]] = 1 + num;
                num++;
            }
        }
    }

    local_matrix = mtx::create(settings.executor);
    local_matrix->copy_from(gko::lend(local_matrix_compute));
    interface_matrix = mtx::create(settings.executor);
    interface_matrix->copy_from(gko::lend(interface_matrix_compute));

    local_matrix->sort_by_column_index();
    interface_matrix->sort_by_column_index();
}


template <typename ValueType, typename IndexType>
void SolverRAS<ValueType, IndexType>::setup_comm_buffers()
{
    using vec_itype = gko::Array<IndexType>;
    using vec_vtype = gko::matrix::Dense<ValueType>;
    using vec_request = gko::Array<MPI_Request>;
    using vec_vecshared = gko::Array<IndexType *>;
    auto metadata = this->metadata;
    auto settings = this->settings;
    auto comm_struct = this->comm_struct;
    auto my_rank = metadata.my_rank;
    auto num_subdomains = metadata.num_subdomains;
    auto first_row = metadata.first_row->get_data();
    auto global_to_local = metadata.global_to_local->get_data();
    auto local_to_global = metadata.local_to_global->get_data();
    auto comm_size = metadata.comm_size;

    auto neighbors_in = this->comm_struct.neighbors_in->get_data();
    auto local_neighbors_in = this->comm_struct.local_neighbors_in->get_data();
    auto global_get = this->comm_struct.global_get->get_data();
    auto is_local_neighbor = this->comm_struct.is_local_neighbor;

    this->comm_struct.num_neighbors_in = 0;
    int num_recv = 0;
    std::vector<int> recv(num_subdomains, 0);
    for (auto p = 0; p < num_subdomains; p++) {
        if (p != my_rank) {
            int count = 0;
            for (auto i = first_row[p]; i < first_row[p + 1]; i++) {
                if (global_to_local[i] != 0) {
                    count++;
                }
            }
            if (count > 0) {
                int pp = this->comm_struct.num_neighbors_in;
                global_get[pp] = new IndexType[1 + count];
                (global_get[pp])[0] = 0;
                for (auto i = first_row[p]; i < first_row[p + 1]; i++) {
                    if (global_to_local[i] != 0) {
                        // global index
                        (global_get[pp])[1 + (global_get[pp])[0]] = i;
                        (global_get[pp])[0]++;
                    }
                }
                neighbors_in[pp] = p;
                this->comm_struct.num_neighbors_in++;

                num_recv += (global_get[pp])[0];
                recv[p] = 1;
            }
            is_local_neighbor[p] =
                Utils<ValueType, IndexType>::check_subd_locality(MPI_COMM_WORLD,
                                                                 p, my_rank);
        }
    }

    std::vector<MPI_Request> send_req1(comm_size);
    std::vector<MPI_Request> send_req2(comm_size);
    std::vector<MPI_Request> recv_req1(comm_size);
    std::vector<MPI_Request> recv_req2(comm_size);
    int zero = 0;
    int pp = 0;
    std::vector<int> send(num_subdomains, 0);

    auto mpi_itype = boost::mpi::get_mpi_datatype(global_get[pp][0]);
    for (auto p = 0; p < num_subdomains; p++) {
        if (p != my_rank) {
            if (recv[p] != 0) {
                MPI_Isend((global_get[pp]), 1, mpi_itype, p, 1, MPI_COMM_WORLD,
                          &send_req1[p]);
                MPI_Isend((global_get[pp]), 1 + (global_get[pp])[0], mpi_itype,
                          p, 2, MPI_COMM_WORLD, &send_req2[p]);
                pp++;
            } else {
                MPI_Isend(&zero, 1, mpi_itype, p, 1, MPI_COMM_WORLD,
                          &send_req1[p]);
            }
            MPI_Irecv(&send[p], 1, mpi_itype, p, 1, MPI_COMM_WORLD,
                      &recv_req1[p]);
        }
    }
    this->comm_struct.num_neighbors_out = 0;
    auto neighbors_out = this->comm_struct.neighbors_out->get_data();
    auto local_neighbors_out =
        this->comm_struct.local_neighbors_out->get_data();
    auto global_put = this->comm_struct.global_put->get_data();
    mpi_itype = boost::mpi::get_mpi_datatype((global_put[pp])[0]);
    int pflag = 0;
    pp = 0;
    int num_send = 0;
    for (auto p = 0; p < num_subdomains; p++) {
        if (p != my_rank) {
            MPI_Status status;
            MPI_Wait(&recv_req1[p], &status);

            if (send[p] > 0) {
                neighbors_out[pp] = p;

                global_put[pp] = new IndexType[1 + send[p]];
                (global_put[pp])[0] = send[p];

                MPI_Irecv((global_put[pp]), 1 + send[p], mpi_itype, p, 2,
                          MPI_COMM_WORLD, &recv_req2[p]);
                MPI_Wait(&recv_req2[p], &status);

                num_send += send[p];
                pp++;
            }
            MPI_Wait(&send_req1[p], &status);
            if (recv[p] != 0) {
                MPI_Wait(&send_req2[p], &status);
            }
        }
    }
    this->comm_struct.num_neighbors_out = pp;

    // allocate MPI buffer
    // one-sided
    if (settings.comm_settings.enable_onesided) {
        if (num_recv > 0) {
            this->comm_struct.recv_buffer =
                vec_vtype::create(settings.executor, gko::dim<2>(num_recv, 1));

            MPI_Win_create(this->comm_struct.recv_buffer->get_values(),
                           num_recv * sizeof(ValueType), sizeof(ValueType),
                           MPI_INFO_NULL, MPI_COMM_WORLD,
                           &(this->comm_struct.window_recv_buffer));
            this->comm_struct.windows_from = std::shared_ptr<vec_itype>(
                new vec_itype(settings.executor->get_master(),
                              this->comm_struct.num_neighbors_in),
                std::default_delete<vec_itype>());
            for (auto j = 0; j < this->comm_struct.num_neighbors_in; j++) {
                // j-th neighbor mapped to j-th window
                this->comm_struct.windows_from->get_data()[j] = j;
            }
        } else {
            this->comm_struct.recv_buffer =
                vec_vtype::create(settings.executor, gko::dim<2>(1, 1));
        }
    }
    // two-sided
    else {
        if (num_recv > 0) {
            this->comm_struct.recv_buffer =
                vec_vtype::create(settings.executor, gko::dim<2>(num_recv, 1));
        } else {
            this->comm_struct.recv_buffer = nullptr;
        }

        this->comm_struct.put_request = std::shared_ptr<vec_request>(
            new vec_request(settings.executor->get_master(),
                            this->comm_struct.num_neighbors_out),
            std::default_delete<vec_request>());
        this->comm_struct.get_request = std::shared_ptr<vec_request>(
            new vec_request(settings.executor->get_master(),
                            this->comm_struct.num_neighbors_in),
            std::default_delete<vec_request>());
    }

    // one-sided
    if (settings.comm_settings.enable_onesided) {
        if (num_send > 0) {
            this->comm_struct.send_buffer =
                vec_vtype::create(settings.executor, gko::dim<2>(num_send, 1));
            MPI_Win_create(this->comm_struct.send_buffer->get_values(),
                           num_send * sizeof(ValueType), sizeof(ValueType),
                           MPI_INFO_NULL, MPI_COMM_WORLD,
                           &(this->comm_struct.window_send_buffer));
            this->comm_struct.windows_to = std::shared_ptr<vec_itype>(
                new vec_itype(settings.executor->get_master(),
                              this->comm_struct.num_neighbors_out),
                std::default_delete<vec_itype>());
            for (auto j = 0; j < this->comm_struct.num_neighbors_out; j++) {
                this->comm_struct.windows_to->get_data()[j] =
                    j;  // j-th neighbor maped to j-th window
            }
        } else {
            this->comm_struct.send_buffer =
                vec_vtype::create(settings.executor, gko::dim<2>(1, 1));
        }
    }
    // two-sided
    else {
        if (num_send > 0) {
            this->comm_struct.send_buffer =
                vec_vtype::create(settings.executor, gko::dim<2>(num_send, 1));
        } else {
            this->comm_struct.send_buffer = nullptr;
        }
    }
    this->comm_struct.local_put = this->comm_struct.global_put;
    this->comm_struct.local_get = this->comm_struct.global_get;
    this->comm_struct.remote_put = this->comm_struct.global_put;
    this->comm_struct.remote_get = this->comm_struct.global_get;
}


template <typename ValueType, typename IndexType>
void SolverRAS<ValueType, IndexType>::setup_windows(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &main_buffer)
{
    using vec_itype = gko::Array<IndexType>;
    using vec_vtype = gko::matrix::Dense<ValueType>;
    auto num_subdomains = metadata.num_subdomains;
    auto local_size_o = metadata.local_size_o;
    auto neighbors_in = this->comm_struct.neighbors_in->get_data();
    auto global_get = this->comm_struct.global_get->get_data();
    auto neighbors_out = this->comm_struct.neighbors_out->get_data();
    auto global_put = this->comm_struct.global_put->get_data();

    // set displacement for the MPI buffer
    auto get_displacements = this->comm_struct.get_displacements->get_data();
    auto put_displacements = this->comm_struct.put_displacements->get_data();
    {
        std::vector<IndexType> tmp_num_comm_elems(num_subdomains + 1, 0);
        tmp_num_comm_elems[0] = 0;
        for (auto j = 0; j < this->comm_struct.num_neighbors_in; j++) {
            if ((global_get[j])[0] > 0) {
                int p = neighbors_in[j];
                tmp_num_comm_elems[p + 1] = (global_get[j])[0];
            }
        }
        for (auto j = 0; j < num_subdomains; j++) {
            tmp_num_comm_elems[j + 1] += tmp_num_comm_elems[j];
        }

        auto mpi_itype = boost::mpi::get_mpi_datatype(tmp_num_comm_elems[0]);
        MPI_Alltoall(tmp_num_comm_elems.data(), 1, mpi_itype, put_displacements,
                     1, mpi_itype, MPI_COMM_WORLD);
    }

    {
        std::vector<IndexType> tmp_num_comm_elems(num_subdomains + 1, 0);
        tmp_num_comm_elems[0] = 0;
        for (auto j = 0; j < this->comm_struct.num_neighbors_out; j++) {
            if ((global_put[j])[0] > 0) {
                int p = neighbors_out[j];
                tmp_num_comm_elems[p + 1] = (global_put[j])[0];
            }
        }
        for (auto j = 0; j < num_subdomains; j++) {
            tmp_num_comm_elems[j + 1] += tmp_num_comm_elems[j];
        }

        auto mpi_itype = boost::mpi::get_mpi_datatype(tmp_num_comm_elems[0]);
        MPI_Alltoall(tmp_num_comm_elems.data(), 1, mpi_itype, get_displacements,
                     1, mpi_itype, MPI_COMM_WORLD);
    }

    // setup windows
    if (settings.comm_settings.enable_onesided) {
        // Onesided
        MPI_Win_create(main_buffer->get_values(),
                       main_buffer->get_size()[0] * sizeof(ValueType),
                       sizeof(ValueType), MPI_INFO_NULL, MPI_COMM_WORLD,
                       &(this->comm_struct.window_x));
    }


    if (settings.comm_settings.enable_onesided) {
        // MPI_Alloc_mem ? Custom allocator ?  TODO
        MPI_Win_create(this->local_residual_vector->get_values(),
                       (num_subdomains) * sizeof(ValueType), sizeof(ValueType),
                       MPI_INFO_NULL, MPI_COMM_WORLD,
                       &(this->window_residual_vector));
        std::vector<IndexType> zero_vec(num_subdomains, 0);
        gko::Array<IndexType> temp_array{settings.executor->get_master(),
                                         zero_vec.begin(), zero_vec.end()};
        this->convergence_vector = std::shared_ptr<vec_itype>(
            new vec_itype(settings.executor->get_master(), temp_array),
            std::default_delete<vec_itype>());
        this->convergence_sent = std::shared_ptr<vec_itype>(
            new vec_itype(settings.executor->get_master(), num_subdomains),
            std::default_delete<vec_itype>());
        this->convergence_local = std::shared_ptr<vec_itype>(
            new vec_itype(settings.executor->get_master(), num_subdomains),
            std::default_delete<vec_itype>());
        MPI_Win_create(this->convergence_vector->get_data(),
                       (num_subdomains) * sizeof(IndexType), sizeof(IndexType),
                       MPI_INFO_NULL, MPI_COMM_WORLD,
                       &(this->window_convergence));
    }

    if (settings.comm_settings.enable_onesided && num_subdomains > 1) {
        // Lock all windows.
        if (settings.comm_settings.enable_get &&
            settings.comm_settings.enable_lock_all) {
            MPI_Win_lock_all(0, this->comm_struct.window_send_buffer);
        }
        if (settings.comm_settings.enable_put &&
            settings.comm_settings.enable_lock_all) {
            MPI_Win_lock_all(0, this->comm_struct.window_recv_buffer);
        }
        if (settings.comm_settings.enable_one_by_one &&
            settings.comm_settings.enable_lock_all) {
            MPI_Win_lock_all(0, this->comm_struct.window_x);
        }
        MPI_Win_lock_all(0, this->window_residual_vector);
        MPI_Win_lock_all(0, this->window_convergence);
    }
}


template <typename ValueType, typename IndexType>
void exchange_boundary_onesided(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType>::comm_struct &comm_struct,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &work_vector,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &cpu_work_vector,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution)
{
    using vec_vtype = gko::matrix::Dense<ValueType>;
    using arr = gko::Array<IndexType>;
    using varr = gko::Array<ValueType>;

    auto num_neighbors_out = comm_struct.num_neighbors_out;
    auto local_num_neighbors_out = comm_struct.local_num_neighbors_out;
    auto neighbors_out = comm_struct.neighbors_out->get_const_data();
    auto local_neighbors_out =
        comm_struct.local_neighbors_out->get_const_data();
    auto global_put = comm_struct.global_put->get_data();
    auto local_put = comm_struct.local_put->get_data();
    auto remote_put = comm_struct.remote_put->get_data();
    auto put_displacements = comm_struct.put_displacements->get_const_data();
    auto send_buffer = comm_struct.send_buffer->get_values();

    auto num_neighbors_in = comm_struct.num_neighbors_in;
    auto local_num_neighbors_in = comm_struct.local_num_neighbors_in;
    auto neighbors_in = comm_struct.neighbors_in->get_const_data();
    auto local_neighbors_in = comm_struct.local_neighbors_in->get_const_data();
    auto global_get = comm_struct.global_get->get_data();
    auto local_get = comm_struct.local_get->get_data();
    auto remote_get = comm_struct.remote_get->get_data();
    auto get_displacements = comm_struct.get_displacements->get_const_data();
    auto recv_buffer = comm_struct.recv_buffer->get_values();
    // bool* is_local_neighbor = comm_struct.is_local_neighbor.data();

    ValueType dummy = 1.0;
    auto mpi_vtype = boost::mpi::get_mpi_datatype(dummy);
    if (settings.comm_settings.enable_put) {
        if (settings.comm_settings.enable_one_by_one) {
            CommHelpers::transfer_one_by_one(
                settings, comm_struct.window_x, global_solution->get_values(),
                global_put, num_neighbors_out, neighbors_out);
        } else {
            int num_put = 0;
            for (auto p = 0; p < num_neighbors_out; p++) {
                // send
                if ((global_put[p])[0] > 0) {
                    CommHelpers::pack_buffer(
                        settings, global_solution->get_values(), send_buffer,
                        work_vector->get_values(), global_put, num_put, p,
                        comm_struct.is_local_neighbor);
                    CommHelpers::transfer_buffer(
                        settings, comm_struct.window_recv_buffer, send_buffer,
                        global_put, num_put, p, neighbors_out,
                        put_displacements, comm_struct.is_local_neighbor);
                    num_put += (global_put[p])[0];
                }
            }
        }
        // unpack receive buffer
        int num_get = 0;
        for (auto p = 0; p < num_neighbors_in; p++) {
            if ((global_get[p])[0] > 0) {
                CommHelpers::unpack_buffer(
                    settings, global_solution->get_values(), recv_buffer,
                    work_vector->get_values(), global_get, num_get, p,
                    comm_struct.is_local_neighbor);
                num_get += (global_get[p])[0];
            }
        }
    } else if (settings.comm_settings.enable_get) {
        if (settings.comm_settings.enable_one_by_one) {
            CommHelpers::transfer_one_by_one(
                settings, comm_struct.window_x, global_solution->get_values(),
                global_get, num_neighbors_in, neighbors_in);
        } else {
            // Gather into send buffer so that the procs can Get from it
            int num_put = 0;
            for (auto p = 0; p < num_neighbors_out; p++) {
                if ((global_put[p])[0] > 0) {
                    CommHelpers::pack_buffer(
                        settings, global_solution->get_values(), send_buffer,
                        work_vector->get_values(), global_put, num_put, p,
                        comm_struct.is_local_neighbor);
                }
                num_put += (global_put[p])[0];
            }
            int num_get = 0;
            for (auto p = 0; p < num_neighbors_in; p++) {
                if ((global_get[p])[0] > 0) {
                    CommHelpers::transfer_buffer(
                        settings, comm_struct.window_send_buffer, recv_buffer,
                        global_get, num_get, p, neighbors_in, get_displacements,
                        comm_struct.is_local_neighbor);
                    CommHelpers::unpack_buffer(
                        settings, global_solution->get_values(), recv_buffer,
                        work_vector->get_values(), global_get, num_get, p,
                        comm_struct.is_local_neighbor);
                }
                num_get += (global_get[p])[0];
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void exchange_boundary_twosided(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType>::comm_struct &comm_struct,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution)
{
    using vec = gko::matrix::Dense<ValueType>;
    MPI_Status status;
    auto num_neighbors_out = comm_struct.num_neighbors_out;
    auto neighbors_out = comm_struct.neighbors_out->get_data();
    auto global_put = comm_struct.global_put->get_data();
    auto local_put = comm_struct.local_put->get_data();
    auto put_request = comm_struct.put_request->get_data();
    int num_put = 0;
    ValueType dummy = 0.0;
    for (auto p = 0; p < num_neighbors_out; p++) {
        // send
        if ((global_put[p])[0] > 0) {
            if (settings.comm_settings.enable_overlap &&
                metadata.iter_count > 1) {
                // wait for the previous send
                auto p_r = put_request[p];
                MPI_Wait(&p_r, &status);
            }
            auto send_buffer = comm_struct.send_buffer->get_values();
            auto mpi_vtype = boost::mpi::get_mpi_datatype(dummy);
            settings.executor->run(GatherScatter<ValueType, IndexType>(
                true, (global_put[p])[0], &((local_put[p])[1]),
                global_solution->get_values(), &send_buffer[num_put]));

            MPI_Isend(&send_buffer[num_put], (global_put[p])[0], mpi_vtype,
                      neighbors_out[p], 0, MPI_COMM_WORLD, &put_request[p]);
            num_put += (global_put[p])[0];
        }
    }

    int num_get = 0;
    auto get_request = comm_struct.get_request->get_data();
    auto num_neighbors_in = comm_struct.num_neighbors_in;
    auto neighbors_in = comm_struct.neighbors_in->get_data();
    auto global_get = comm_struct.global_get->get_data();
    auto local_get = comm_struct.local_get->get_data();
    if (!settings.comm_settings.enable_overlap || metadata.iter_count == 0) {
        for (auto p = 0; p < num_neighbors_in; p++) {
            // receive
            if ((global_get[p])[0] > 0) {
                auto recv_buffer = comm_struct.recv_buffer->get_values();
                auto mpi_vtype = boost::mpi::get_mpi_datatype(recv_buffer[0]);
                MPI_Irecv(&recv_buffer[num_get], (global_get[p])[0], mpi_vtype,
                          neighbors_in[p], 0, MPI_COMM_WORLD, &get_request[p]);
                num_get += (global_get[p])[0];
            }
        }
    }

    num_get = 0;
    // wait for receive
    for (auto p = 0; p < num_neighbors_in; p++) {
        if ((global_get[p])[0] > 0) {
            auto recv_buffer = comm_struct.recv_buffer->get_values();
            auto mpi_vtype = boost::mpi::get_mpi_datatype(recv_buffer[0]);
            settings.executor->run(GatherScatter<ValueType, IndexType>(
                false, (global_get[p])[0], &((local_get[p])[1]),
                &recv_buffer[num_get], global_solution->get_values()));

            if (settings.comm_settings.enable_overlap) {
                // start the next receive
                MPI_Irecv(&recv_buffer[num_get], (global_get[p])[0], mpi_vtype,
                          neighbors_in[p], 0, MPI_COMM_WORLD, &get_request[p]);
            }
            num_get += (global_get[p])[0];
        }
    }

    // wait for send
    if (!settings.comm_settings.enable_overlap) {
        for (auto p = 0; p < num_neighbors_out; p++) {
            if ((global_put[p])[0] > 0) {
                auto p_r = put_request[p];
                MPI_Wait(&p_r, &status);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void SolverRAS<ValueType, IndexType>::exchange_boundary(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution)
{
    if (settings.comm_settings.enable_onesided) {
        exchange_boundary_onesided<ValueType, IndexType>(
            settings, metadata, this->comm_struct, this->work_vector,
            this->cpu_work_vector, global_solution);
    } else {
        exchange_boundary_twosided<ValueType, IndexType>(
            settings, metadata, this->comm_struct, global_solution);
    }
}


template <typename ValueType, typename IndexType>
void SolverRAS<ValueType, IndexType>::update_boundary(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &interface_matrix)
{
    using vec_vtype = gko::matrix::Dense<ValueType>;
    auto one = gko::initialize<gko::matrix::Dense<ValueType>>(
        {1.0}, settings.executor);
    auto neg_one = gko::initialize<gko::matrix::Dense<ValueType>>(
        {-1.0}, settings.executor);
    auto local_size_x = metadata.local_size_x;
    local_solution->copy_from(local_rhs.get());
    if (metadata.num_subdomains > 1 && settings.overlap > 0) {
        auto temp_solution = vec_vtype::create(
            settings.executor, local_solution->get_size(),
            gko::Array<ValueType>::view(settings.executor,
                                        local_solution->get_size()[0],
                                        global_solution->get_values()),
            1);
        interface_matrix->apply(neg_one.get(), temp_solution.get(), one.get(),
                                local_solution.get());
    }
}


#define DECLARE_SOLVER_RAS(ValueType, IndexType) \
    class SolverRAS<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_SOLVER_RAS);
#undef DECLARE_SOLVER_RAS


}  // namespace schwz
