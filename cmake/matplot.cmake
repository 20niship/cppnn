include(FetchContent)
FetchContent_Declare(
  matplotlib
  GIT_REPOSITORY  https://github.com/lava/matplotlib-cpp.git
  GIT_TAG        origin/master
)

message("Downloading external libraries......")
FetchContent_MakeAvailable(matplotlib)
