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

#ifndef communicate_hpp
#define communicate_hpp


#include <memory>


#include <mpi.h>


#include "initialization.hpp"


namespace schwz {


/**
 * The communication class that provides the methods for the communication
 * between the subdomains.
 *
 * @tparam ValueType  The type of the floating point values.
 * @tparam IndexType  The type of the index type values.
 *
 * @ref comm
 * @ingroup comm
 */
template <typename ValueType, typename IndexType, typename MixedValueType>
class Communicate {
public:
    friend class Initialize<ValueType, IndexType>;

    /**
     * The communication struct used to store the communication data.
     */
    struct comm_struct {
    public:
        /**
         * The number of neighbors this subdomain has to receive data from.
         */
        int num_neighbors_in;

        /**
         * The number of neighbors this subdomain has to send data to.
         */
        int num_neighbors_out;

        /**
         * The neighbors this subdomain has to receive data from.
         */
        std::shared_ptr<gko::Array<IndexType>> neighbors_in;

        /**
         * The neighbors this subdomain has to send data to.
         */
        std::shared_ptr<gko::Array<IndexType>> neighbors_out;

        /**
         * The bool vector which is true if the neighbors of a subdomain are in
         * one node..
         */
        std::vector<bool> is_local_neighbor;
        /**
         * The number of neighbors this subdomain has to receive data from.
         */
        int local_num_neighbors_in;

        /**
         * The number of neighbors this subdomain has to send data to.
         */
        int local_num_neighbors_out;

        /**
         * The neighbors this subdomain has to receive data from.
         */
        std::shared_ptr<gko::Array<IndexType>> local_neighbors_in;

        /**
         * The neighbors this subdomain has to send data to.
         */
        std::shared_ptr<gko::Array<IndexType>> local_neighbors_out;

        /**
         * The array containing the number of elements that each subdomain sends
         * from the other. For example. global_put[p][0] contains the overall
         * number of elements to be sent to subdomain p and global_put[p][i]
         * contains the index of the solution vector to be sent to subdomain p.
         */
        std::shared_ptr<gko::Array<IndexType *>> global_put;

        /**
         * @copydoc global_put
         */
        std::shared_ptr<gko::Array<IndexType *>> local_put;

        /**
         * The array containing the number of elements that each subdomain gets
         * from the other. For example. global_get[p][0] contains the overall
         * number of elements to be received to subdomain p and global_put[p][i]
         * contains the index of the solution vector to be received from
         * subdomain p.
         */
        std::shared_ptr<gko::Array<IndexType *>> global_get;

        /**
         * @copydoc global_get
         */
        std::shared_ptr<gko::Array<IndexType *>> local_get;

        /**
         * The number of elements being sent to each subdomain.
         */
        std::vector<IndexType> send;

        /**
         * The number of elements being sent to each subdomain.
         */
        std::vector<IndexType> recv;

        /**
         * The RDMA window ids.
         */
        std::shared_ptr<gko::Array<IndexType>> window_ids;

        /**
         * The RDMA window ids to receive data from.
         */
        std::shared_ptr<gko::Array<IndexType>> windows_from;

        /**
         * The RDMA window ids to send data to.
         */
        std::shared_ptr<gko::Array<IndexType>> windows_to;

        /**
         * The put request array.
         */
        std::shared_ptr<gko::Array<MPI_Request>> put_request;

        /**
         * The get request array.
         */
        std::shared_ptr<gko::Array<MPI_Request>> get_request;

        /**
         * The send buffer used for the actual communication for both one-sided
         * and two-sided (always allocated).
         */
        std::shared_ptr<gko::matrix::Dense<ValueType>> send_buffer;

        /**
         * The mixed send buffer used for the actual communication for both
         * one-sided and two-sided.
         */
        std::shared_ptr<gko::matrix::Dense<MixedValueType>> mixedt_send_buffer;

        /**
         * The recv buffer used for the actual communication for both one-sided
         * and two-sided (always allocated).
         */
        std::shared_ptr<gko::matrix::Dense<ValueType>> recv_buffer;

        /**
         * The mixed precision recv buffer used for the actual communication for
         * both one-sided and two-sided.
         */
        std::shared_ptr<gko::matrix::Dense<MixedValueType>> mixedt_recv_buffer;

        /**
         * The displacements for the receiving of the buffer.
         */
        std::shared_ptr<gko::Array<IndexType>> get_displacements;

        /**
         * The displacements for the sending of the buffer.
         */
        std::shared_ptr<gko::Array<IndexType>> put_displacements;

        /**
         * The RDMA window for the recv buffer.
         */
        MPI_Win window_recv_buffer;

        /**
         * The RDMA window for the send buffer.
         */
        MPI_Win window_send_buffer;

        /**
         * The RDMA window for the solution vector.
         */
        MPI_Win window_x;
    };
    comm_struct comm_struct;

    /**
     * Sets up the communication buffers needed for the boundary exchange.
     */
    virtual void setup_comm_buffers() = 0;

    /**
     * Sets up the windows needed for the asynchronous communication.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param main_buffer  The main buffer being exchanged between the
     *                     subdomains.
     */
    virtual void setup_windows(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &main_buffer) = 0;

    /**
     * Exchanges the elements of the solution vector
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param global_solution  The solution vector being exchanged between the
     *                     subdomains.
     */
    virtual void exchange_boundary(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution) = 0;

    /**
     * Transforms data from a local vector to a global vector
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param local_vector  The local vector in question.
     * @param global_vector  The global vector in question.
     */
    void local_to_global_vector(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_vector,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &global_vector);

    /**
     * Update the values into local vector from obtained from the neighboring
     * sub-domains using the interface matrix.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param local_solution  The local solution vector in the subdomain.
     * @param local_rhs  The local right hand side vector in the subdomain.
     * @param global_solution  The workspace solution vector.
     * @param global_old_solution  The global solution vector of the previous
     *                             iteration.
     * @param interface_matrix The interface matrix containing the interface and
     *                         the overlap data mainly used for exchanging
     *                         values between different sub-domains.
     */
    virtual void update_boundary(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs,
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &interface_matrix) = 0;

    /**
     * Clears the data.
     */
    void clear(Settings &settings);
};


}  // namespace schwz


#endif  // communicate.hpp
