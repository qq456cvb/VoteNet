#!/usr/bin/env bash

/usr/local/cuda/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')  -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -L$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
#
# TF1.4
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') -I /usr/local/cuda/include -I $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
