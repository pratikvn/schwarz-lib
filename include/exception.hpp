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
      : what_(file + ":" + std::to_string(line) + ": " + what) {}

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
      : Error(file, line, func + " is not implemented") {}
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
      : Error(file, line, func + ": " + get_error(error_code)) {}

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
      : Error(file, line, func + ": " + get_error(error_code)) {}

private:
  static std::string get_error(int error_code);
};

#endif
/*----------------------------   exception.hpp ---------------------------*/
