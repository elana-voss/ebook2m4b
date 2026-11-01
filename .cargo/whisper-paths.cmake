# Read by whisper-rs-sys's CMake build (CMAKE_PROJECT_INCLUDE in config.toml) at each project() of whisper.cpp:
# ggml's Vulkan shader generator is an ExternalProject, whose default folders under the crate's OUT_DIR make
# MSBuild's paths pass Windows' 260 characters; EP_BASE puts its folders at <build>/ep/Build/<name> instead.
# Trap: whisper-rs-sys hands rustc every folder under its build as a link path, and one too long makes rustc
# "can't find crate" (seen for tracing); after such a failed build, `cargo clean -p whisper-rs-sys`.
set_property(DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY EP_BASE "${CMAKE_BINARY_DIR}/ep")
