#!/usr/bin/env bash
# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') -I /usr/local/cuda/include -I $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
