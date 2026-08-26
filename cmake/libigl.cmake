if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    # Temporarily off libigl/libigl#2553 (branch off main) rather than v2.5.0,
    # to drop the macOS Mojave hide/show hack that flashes the viewer window on
    # launch. Move back to a tag once that lands.
    GIT_TAG 0d5e06142c89cb2b7c77a93d6c484351d5a31ec0
)
FetchContent_MakeAvailable(libigl)
