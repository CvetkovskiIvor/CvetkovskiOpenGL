# komentar
add_definitions(
        -DTW_STATIC
        -DTW_NO_LIB_PRAGMA
        -DTW_NO_DIRECT3D
        -DGLEW_STATIC
        -D_CRT_SECURE_NO_WARNINGS
)


set(CMAKE_CXX_STANDARD 17) # ako javlja gresku, stavite 14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
configure_file (
                "${PROJECT_SOURCE_DIR}/studenti/cvetkovski/pathconfig.h.in"
                "${PROJECT_BINARY_DIR}/cvetkovski/pathconfig.h"
)

##########################################
## ovdje navedite svoje datoteke
add_executable(cvetkovski 
            # ovdje navesti svoje datoteke
            cvetkovski.cpp
            sphere.cpp sphere.h
            shader.cpp shader.h
            vertexbuffer.cpp vertexbuffer.h
            indexbuffer.cpp indexbuffer.h
            vertexbufferlayout.cpp vertexbufferlayout.h
            vertexarray.cpp vertexarray.h
            texture.cpp texture.h
            sphere.cpp sphere.h
            line.cpp line.h
            matrices.cpp matrices.h
            renderer.cpp renderer.h
            )

target_include_directories(cvetkovski PRIVATE
            ${PROJECT_SOURCE_DIR}/third_party/glfw-3.3.2/include/
            ${PROJECT_SOURCE_DIR}/third_party/glm/
            ${PROJECT_SOURCE_DIR}/third_party/glew-2.1.0/include/
            ${PROJECT_SOURCE_DIR}/third_party/stb
            ${PROJECT_BINARY_DIR}
            ${PROJECT_BINARY_DIR}/cvetkovski/
)

target_link_libraries(cvetkovski
        ${OPENGL_LIBRARY}
        glfw
        GLEW_2100
)

set_property(TARGET cvetkovski PROPERTY CXX_STANDARD 17)
set_property(TARGET cvetkovski PROPERTY CXX_STANDARD_REQUIRED ON)
