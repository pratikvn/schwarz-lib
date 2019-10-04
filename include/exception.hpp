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
