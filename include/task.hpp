
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


#include <deque>
#include <exception>
#include <future>
#include <thread>
#include <vector>

#include <schwarz/config.hpp>

#if SCHW_HAVE_CUDA
#include <cuda_runtime.h>
#endif


#include <exception_helpers.hpp>


namespace SchwarzWrappers {


class Task {
public:
    template <class F, class R = typename std::result_of<F &()>::type>
    std::future<R> queue(F &&f)
    {
        std::packaged_task<R()> p(std::forward<F>(f));

        auto r = p.get_future();
        {
            std::unique_lock<std::mutex> l(m);
            work.emplace_back(std::move(p));
        }
        v.notify_one();

        return r;
    }

    void start(std::size_t N = 1)
    {
        for (std::size_t i = 0; i < N; ++i) {
            finished.push_back(
                std::async(std::launch::async, [this] { thread_task(); }));
        }
    }
    void abort()
    {
        cancel_pending();
        finish();
    }
    void cancel_pending()
    {
        std::unique_lock<std::mutex> l(m);
        work.clear();
    }
    void finish()
    {
        {
            std::unique_lock<std::mutex> l(m);
            for (auto &&unused : finished) {
                work.push_back({});
            }
        }
        v.notify_all();
        finished.clear();
    }
    ~Task() { finish(); }

    std::mutex m;
    std::condition_variable v;
    std::deque<std::packaged_task<void()>> work;
    std::vector<std::future<void>> finished;

private:
    void thread_task()
    {
        while (true) {
            std::packaged_task<void()> f;
            {
                std::unique_lock<std::mutex> l(m);
                if (work.empty()) {
                    v.wait(l, [&] { return !work.empty(); });
                }
                f = std::move(work.front());
                work.pop_front();
            }
            if (!f.valid()) return;
            f();
        }
    }
};


}  // namespace SchwarzWrappers


#endif  // SCHWARZ_DEVICE_GUARD_HPP_
