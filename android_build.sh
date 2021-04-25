
export PATH=/Users/xiachunwei/Projects/aarch64-linux-android-ndk-r18b/bin/:$PATH
# CXX=clang++ CC=clang scons arch=arm64-v8a os=android neon=1 tracing=1 build_dir=./build opencl=1 -j6
CXX=clang++ CC=clang scons Werror=1 -j1 debug=0 asserts=1 neon=1 opencl=1 embed_kernels=1 os=android arch=arm64-v8a build_dir=release_build
