#!/bin/sh
# Self-contained arm64 niimath for the macOS package: OpenMP from a static libomp built for macOS 13
# (Homebrew's libomp.a targets the build machine's macOS), no zstd, only system libraries.
#   packaging/build_niimath.sh   ->  target/niimath/niimath
# NIIMATH_SRC: a niimath checkout (default: clone rordenlab/niimath at NIIMATH_COMMIT).
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work="$root/target"
llvm_version=21.1.8
niimath_commit=${NIIMATH_COMMIT:-7a87e6f9e03d10ccb4f798639071fc58acaecf1f}
mkdir -p "$work/llvm"
if [ ! -f "$work/llvm/install/lib/libomp.a" ]; then
	cd "$work/llvm"
	# The runtime is linked into the signed package: pin the sources.
	for p in "openmp 856b023748b41ac7b2c83fd8e9f765ff48a4df2fe6777d2811ef7c7ed8f2f977" "cmake 85735f20fd8c81ecb0a09abb0c267018475420e93b65050cc5b7634eab744de9"; do
		set -- $p
		[ -d "$1-$llvm_version.src" ] && continue
		curl -fsSL -o "$1.tar.xz" "https://github.com/llvm/llvm-project/releases/download/llvmorg-$llvm_version/$1-$llvm_version.src.tar.xz"
		echo "$2  $1.tar.xz" | shasum -a 256 -c - >/dev/null || { echo "build_niimath.sh: $1.tar.xz checksum mismatch" >&2; exit 1; }
		tar xJf "$1.tar.xz"
	done
	[ -d cmake ] || mv "cmake-$llvm_version.src" cmake
	cmake -S "openmp-$llvm_version.src" -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
		-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DLIBOMP_ENABLE_SHARED=OFF -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
		-DLIBOMP_OMPT_SUPPORT=OFF -DCMAKE_INSTALL_PREFIX="$work/llvm/install" >/dev/null
	cmake --build build -j 8 >/dev/null
	cmake --install build >/dev/null
fi
src=${NIIMATH_SRC:-$work/niimath-src}
if [ -z "${NIIMATH_SRC:-}" ]; then
	[ -d "$src" ] || git clone --quiet https://github.com/rordenlab/niimath "$src"
	git -C "$src" fetch --quiet origin
	git -C "$src" checkout --quiet "$niimath_commit"
fi
omp="$work/llvm/install"
# The superbuild does not forward OpenMP hints or the deployment target, so configure src/ directly.
cmake -S "$src/src" -B "$work/niimath" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
	-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DENABLE_ZSTD=OFF -DUSE_OPENMP=ON \
	"-DOpenMP_C_FLAGS=-Xpreprocessor -fopenmp -I$omp/include" -DOpenMP_C_LIB_NAMES=omp \
	-DOpenMP_omp_LIBRARY="$omp/lib/libomp.a" >/dev/null
cmake --build "$work/niimath" -j 8 >/dev/null
"$work/niimath/niimath" 2>&1 | head -1
