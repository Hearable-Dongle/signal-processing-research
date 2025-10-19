# Start with the official Miniconda3 image
FROM continuumio/miniconda3

# Set environment variables
ENV TZ=UTC
ENV DEBIAN_FRONTEND=noninteractive

# --- 1. INSTALL SYSTEM DEPENDENCIES (Including .NET) ---
RUN apt-get update && apt-get install -y \
        git build-essential cmake ninja-build protobuf-compiler libprotobuf-dev \
        nlohmann-json3-dev xxd file wget gpg apt-transport-https pybind11-dev \
    && \
    echo ">>> Installing .NET SDK and Host..." && \
    wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y dotnet-sdk-6.0 dotnet-host && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- 2. MANUALLY CREATE THE MISSING NETHOST CMAKE TARGET FILE ---
RUN echo ">>> Manually creating nethost-config.cmake..." && \
    NETHOST_LIB_PATH=$(find /usr/share/dotnet -name "libnethost.so") && \
    NETHOST_INCLUDE_DIR=$(dirname "$(find /usr/share/dotnet -name 'nethost.h')") && \
    mkdir -p /usr/local/lib/cmake/nethost && \
    printf "if(NOT TARGET nethost::nethost)\n    add_library(nethost::nethost UNKNOWN IMPORTED)\n    set_target_properties(nethost::nethost PROPERTIES\n        IMPORTED_LOCATION \"${NETHOST_LIB_PATH}\"\n        INTERFACE_INCLUDE_DIRECTORIES \"${NETHOST_INCLUDE_DIR}\"\n    )\nendif()\n" > /usr/local/lib/cmake/nethost/nethost-config.cmake && \
    echo ">>> nethost-config.cmake created successfully." && \
    cat /usr/local/lib/cmake/nethost/nethost-config.cmake


# # --- 2. MANUALLY CREATE THE MISSING NETHOST CMAKE FILE ---
# RUN echo ">>> Manually creating nethost-config.cmake..." && \
#     # Find the required .NET paths
#     NETHOST_HEADER_PATH=$(find /usr/share/dotnet -name "nethost.h") && \
#     NETHOST_LIB_PATH=$(find /usr/share/dotnet -name "libnethost.so") && \
#     NETHOST_INCLUDE_DIR=$(dirname "${NETHOST_HEADER_PATH}") && \
#     # Create the cmake config directory
#     mkdir -p /usr/local/lib/cmake/nethost && \
#     # Write the config file
#     echo 'set(nethost_INCLUDE_DIRS "'${NETHOST_INCLUDE_DIR}'")' > /usr/local/lib/cmake/nethost/nethost-config.cmake && \
#     echo 'set(nethost_LIBRARIES "'${NETHOST_LIB_PATH}'")' >> /usr/local/lib/cmake/nethost/nethost-config.cmake && \
#     echo ">>> nethost-config.cmake created successfully." && \
#     cat /usr/local/lib/cmake/nethost/nethost-config.cmake

# --- 2b. INSTALL HALIDE DEPENDENCY ---
# ENV HALIDE_VERSION=14.0.0
# ENV HALIDE_DISTRIB_PATH=/opt/halide
# RUN echo ">>> Installing Halide v${HALIDE_VERSION}..." && \
#     wget https://github.com/halide/Halide/releases/download/v${HALIDE_VERSION}/halide-linux-${HALIDE_VERSION}-x86-64.tar.gz && \
#     mkdir -p ${HALIDE_DISTRIB_PATH} && \
#     tar -xvf halide-linux-${HALIDE_VERSION}-x86-64.tar.gz -C ${HALIDE_DISTRIB_PATH} --strip-components 1 && \
#     rm halide-linux-${HALIDE_VERSION}-x86-64.tar.gz

# --- 3. CREATE AND CONFIGURE CONDA ENVIRONMENT ---
RUN conda create -n nncase_env python=3.10 -y
SHELL ["/bin/bash", "-c"]
RUN conda run -n nncase_env pip install --no-cache-dir torch onnx onnxsim numpy setuptools wheel

# --- 4. BUILD DEPENDENCIES & NNCASE FROM SOURCE ---
RUN source activate nncase_env && \
    \
    # ---- Manually build and install gsl-lite ----
    echo ">>> Building and installing dependency: gsl-lite" && \
    git clone --branch v0.40.0 --depth 1 https://github.com/gsl-lite/gsl-lite.git && \
    cd gsl-lite && mkdir build && cd build && \
    cmake .. && cmake --install . && \
    cd / && rm -rf gsl-lite && \
    \
    # ---- Build nncase ----
    echo ">>> Building and installing nncase from source" && \
    git clone --branch v2.10.0 --depth 1 https://github.com/kendryte/nncase.git && \
    cd nncase && git submodule update --init --recursive && mkdir build && cd build && \
    cmake .. -G Ninja \
    	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_PYTHON_BINDING=ON \
	-DCMAKE_CXX_FLAGS="-mavx -mavx2" && \
    ninja && \
    pip install nncase-kpu && \
    #pip install python/dist/*.whl nncase-kpu && \
    cd / && rm -rf nncase

# --- 5. VERIFY AND FINALIZE ---
#$ RUN echo "Verifying nncase installation..." && \
#     conda run -n nncase_env python -c "import nncase; import nncase.runtime; print('✅ nncase v2.10.0 and runtime verified successfully!')"

WORKDIR /workspace
CMD ["/bin/bash", "--login", "-c", "conda activate nncase_env && /bin/bash"]
