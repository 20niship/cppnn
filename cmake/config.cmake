

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # does not produce the json file
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") # works


# --------------------------------------------------------------
# -----------------    Compile Flags  --------------------------
# --------------------------------------------------------------
# if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
#   message( FATAL_ERROR "Intel IOCのビルドにはまだ対応していないよ")
# elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#   message( FATAL_ERROR "Windowsのビルドにはまだ対応していないよ")
# endif()

include(CheckCXXCompilerFlag)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
check_cxx_compiler_flag("-std=c++20" COMPILER_SUPPORTS_CXX20)
check_cxx_compiler_flag("-std=c++2a" COMPILER_SUPPORTS_CXX2A)
if(${COMPILER_SUPPORTS_CXX20})
  set(CMAKE_CXX_FLAGS "-std=c++20 ${CMAKE_CXX_FLAGS}")
elseif(${COMPILER_SUPPORTS_CXX2A})
  set(CMAKE_CXX_FLAGS "-std=c++2a ${CMAKE_CXX_FLAGS}")
else()
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>")

add_definitions(-w)
  set(CMAKE_CXX_FLAGS "/std:c++latest ${CMAKE_CXX_FLAGS}")
endif()


if (CMAKE_BUILD_TYPE STREQUAL "Release")
  message("Releaseモードでビルドします")
else()
  message("Debugモードでビルドします")
endif()

if (MSVC)
  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "/O2 -DNDEBUG ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS}")
  endif()
else(MSVC)
  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS}")
  endif()
endif (MSVC)


message("Compiler:\n\t${CMAKE_CXX_COMPILER}")
message("compiler flags:\n\t${CMAKE_CXX_FLAGS}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   add_compile_options (-fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   add_compile_options (-fcolor-diagnostics)
endif()
