configure_file("${Schwarz_SOURCE_DIR}/matrices/config.hpp.in"
               "${Schwarz_BINARY_DIR}/matrices/config.hpp" @ONLY)

configure_file("./ani3_crop.mtx" "${Schwarz_BINARY_DIR}/matrices/ani3_crop.mtx")
configure_file("./ani4_crop.mtx" "${Schwarz_BINARY_DIR}/matrices/ani4_crop.mtx")
