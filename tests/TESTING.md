For testing, the Catch2 framework (https://github.com/catchorg/Catch2) is used.

The defines in catch_main.cpp must be used only once and hence is compiled once and linked to every test executable. The directories are self-explanatory.

To add a test, add the name of the source file to the `TEST_EXECS` variable within the required sub-directory, without the extension. This will also be the name of the resulting test executable.
