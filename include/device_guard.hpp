
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


#ifndef SCHWARZ_DEVICE_GUARD_HPP_
#define SCHWARZ_DEVICE_GUARD_HPP_


#include <exception>

#include <schwarz/config.hpp>

#if SCHW_HAVE_CUDA
#include <cuda_runtime.h>
#endif


#include <exception_helpers.hpp>


namespace SchwarzWrappers {


/**
 * This class defines a device guard for the cuda functions and the cuda module.
 * The guard is used to make sure that the device code is run on the correct
 * cuda device, when run with multiple devices. The class records the current
 * device id and uses `cudaSetDevice` to set the device id to the one being
 * passed in. After the scope has been exited, the destructor sets the device_id
 * back to the one before entering the scope.
 */
class device_guard {
public:
    device_guard(int device_id)
    {
#if SCHW_HAVE_CUDA
        SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaGetDevice(&original_device_id));
        SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(device_id));
#endif
    }

    device_guard(device_guard &other) = delete;

    device_guard &operator=(const device_guard &other) = delete;

    device_guard(device_guard &&other) = delete;

    device_guard const &operator=(device_guard &&other) = delete;

    ~device_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (std::uncaught_exception()) {
#if SCHW_HAVE_CUDA
            cudaSetDevice(original_device_id);
#endif
        } else {
#if SCHW_HAVE_CUDA
            SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(original_device_id));
#endif
        }
    }

private:
    int original_device_id{};
};


}  // namespace SchwarzWrappers


#endif  // SCHWARZ_DEVICE_GUARD_HPP_
