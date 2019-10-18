
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

#ifndef exception_hpp
#define exception_hpp


#include <exception>
#include <string>


class Error : public std::exception {
public:
    /**
     * Initializes an error.
     * @param file The name of the offending source file
     * @param line The source code line number where the error occurred
     * @param what The error message
     */
    Error(const std::string &file, int line, const std::string &what)
        : what_(file + ":" + std::to_string(line) + ": " + what)
    {}

    /**
     * Returns a human-readable string with a more detailed description of the
     * error.
     */
    virtual const char *what() const noexcept override { return what_.c_str(); }

private:
    const std::string what_;
};


class NotImplemented : public Error {
public:
    /**
     * Initializes a NotImplemented error.
     * @param file The name of the offending source file
     * @param line The source code line number where the error occurred
     * @param func The name of the not-yet implemented function
     */
    NotImplemented(const std::string &file, int line, const std::string &func)
        : Error(file, line, func + " is not implemented")
    {}
};


class ModuleNotImplemented : public Error {
public:
    /**
     * Initializes a NotImplemented error.
     * @param file The name of the offending source file
     * @param line The source code line number where the error occurred
     * @param module The name of the not-yet implemented module
     * @param func The name of the not-yet implemented function
     */
    ModuleNotImplemented(const std::string &file, int line,
                         const std::string &module, const std::string &func)
        : Error(file, line, module + " in " + func + " is not implemented")
    {}
};


/**
 * BadDimension is thrown if an operation is being applied to a LinOp
 * with bad dimensions.
 */
class BadDimension : public Error {
public:
    /**
     * Initializes a bad dimension error.
     *
     * @param file The name of the offending source file
     * @param line The source code line number where the error occurred
     * @param func The function name where the error occurred
     * @param op_name The name of the operator
     * @param op_num_rows The row dimension of the operator
     * @param op_num_cols The column dimension of the operator
     * @param clarification An additional message further describing the error
     */
    BadDimension(const std::string &file, int line, const std::string &func,
                 const std::string &op_name, std::size_t op_num_rows,
                 std::size_t op_num_cols, const std::string &clarification)
        : Error(file, line,
                func + ": Object " + op_name + " has dimensions [" +
                    std::to_string(op_num_rows) + " x " +
                    std::to_string(op_num_cols) + "]: " + clarification)
    {}
};


/**
 * CudaError is thrown when a CUDA routine throws a non-zero error code.
 */
class CudaError : public Error {
public:
    /**
     * Initializes a CUDA error.
     * @param file The name of the offending source file
     * @param line The source code line number where the error occurred
     * @param func The name of the CUDA routine that failed
     * @param error_code The resulting CUDA error code
     */
    CudaError(const std::string &file, int line, const std::string &func,
              int error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int error_code);
};


/**
 * CusparseError is thrown when a cuSPARSE routine throws a non-zero error code.
 */
class CusparseError : public Error {
public:
    /**
     * Initializes a cuSPARSE error.
     * @param file The name of the offending source file
     * @param line The source code line number where the error occurred
     * @param func The name of the cuSPARSE routine that failed
     * @param error_code The resulting cuSPARSE error code
     */
    CusparseError(const std::string &file, int line, const std::string &func,
                  int error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int error_code);
};


/**
 * MetisError is thrown when a METIS routine throws a non-zero error code.
 */
class MetisError : public Error {
public:
    /**
     * Initializes a METIS error.
     *
     * @param file The name of the offending source file
     * @param line The source code line number where the error occurred
     * @param func The name of the METIS routine that failed
     * @param error_code The resulting METIS error code
     */
    MetisError(const std::string &file, int line, const std::string &func,
               int error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int error_code);
};


#endif  // exception.hpp
