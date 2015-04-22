caffe* parallel 
============================

Overview
============================
caffe* parallel is a faster framework for deep learning, it's forked from 
BVLC/caffe (master branch).(https://github.com/BVLC/caffe ,more details please visit 
http://caffe.berkeleyvision.org).The main achievement of this project is 
data-parallel via MPI.

The source for this version of caffe* parallel can be downloaded from:
https://github.com/sailorsb/caffe-parallel

this version does not support matlab.

Author
============================
Shen,Bo (Inspur) shenbo@inspur.com ; Wang,Yajuan (Inspur) wangyjbj@inspur.com

Changelog:
============================
Ver 0.3alpha:  
 Static assignment task
 GPU DIRECT transfer data

Ver 0.2(20150109):  
 Support LMDB now.(tested mnist)  
 Fixed some bugs.  


Ver 0.1(20141231):

 create project(forked from BVLC/caffe 20141223).
 Data-parallel on levelDB.(Only test cifar10) 
 It's only a simple simple version.We'll as soon as 
possible to improve it and happy new year!

Known Issues:
============================
LMDB
big data(Like ImageNet)
TODO List:
============================
<del>1.support <b>LMDB</b></del>  

2.performance optimization  

3.large-scale test

How to run it
============================
1.Prerequisites
----------------------------
Caffe depends on several software packages.

    CUDA library version(we used 6.0) 6.5, 6.0, 5.5, or 5.0 and the latest driver version for CUDA 6 or 319.* for CUDA 5 (and NOT 331.*)  
    BLAS (we used MKL(14.0.2.144)/ OpenBLAS(r0.2.12))(provided via ATLAS, MKL, or OpenBLAS).  
    OpenCV (we used 2.4.9)(need cmake >=2.8)  
    Boost (we used 1.55)(>= 1.55, although only 1.55 and 1.56 are tested)  
    glog (we used 0.33)  
    gflags (we used 2.1.1)  
    protobuf (we used 2.5.0)  
    protobuf-c  
    leveldb (we used 1.15.0)  
    snappy (we used 1.1.2)  
    hdf5 (we used 1.8.10)  
    lmdb   
    autoconf(>= 2.4)  
    Compiler:  
        g++ compiler(we used 4.4.7)  
    MPI compiler and runtime:  
        Intel MPI (we used 14.0.2.144) / MPICH3 (we used 3.1,CC=gcc,CXX=g++,--enable-threads=multiple)
    For the Python wrapper  
        Python 2.7, numpy (>= 1.7), boost-provided boost.python  

cuDNN Caffe: for fastest operation Caffe is accelerated by drop-in integration of 
NVIDIA cuDNN. To speed up your Caffe models, install cuDNN then uncomment the 
USE_CUDNN := 1 flag in Makefile.config when installing Caffe. Acceleration is 
automatic.  

CPU-only Caffe: for cold-brewed CPU-only Caffe uncomment the CPU_ONLY := 1 flag 
in Makefile.config to configure and build Caffe without CUDA. This is helpful for 
cloud or cluster deployment.  

2.Compile
----------------------------
a. Copy Makefile.config.example and rename Makefile.config  
b. edit Makefile.config:   
   i. If you compile with NVIDIA cuDNN acceleration, you should uncomment the 
USE_CUDNN := 1 flag switch in Makefile.config.  
  
  ii. If there is no GPU in your machine,you should switch tp CPU-only caffe by 
uncommenting the CPU_ONLY := 1 flag in Makefile.config.  
  
 iii. Uncomment CUSTOM_CXX flag and set it : CUSTOM_CXX := mpigxx . If you use Intel 
MPI, please set mpigxx, if you use MPICH3, please set mpicxx, if you use other MPI 
version ,please set the right mpixxx in Makefile.config! (Intel MPI,the default 
compiler is intel compiler； CUDA, should use GNU C++ compiler)  
  iv. Set BLAS: atlas for ATLAS ; mkl for MKL; open for OpenBlas  
  v. Set CUDA_DIR, BLAS_INCLUDE, BLAS_LIB, PYTHON_INCLUDE, PYTHON_LIB,   
INCLUDE_DIRS, LIBRARY_DIRS if you need.  
c. Modify Makefile:  
   i. Add -DMPICH_IGNORE_CXX_SEEK flag to COMMON_FLAGS in "# Debugging" :   
COMMON_FLAGS += -DNDEBUG -O2 -DMPICH_IGNORE_CXX_SEEK  
  ii. Add -mt_mpi flag to CXXFLAGS in "# Complete build flags."(for Intel mpi)  
 iii. Add -mt_mpi flag to LINKFLAGS in "# Complete build flags."(for Intel mpi)  
d. make it.  

3.Run and Test
----------------------------
This program can run 2 processes at least.  
### cifar10  
1. Run data/cifar10/get_cifar10.sh to get cifar10 data.  
2. Run examples/cifar10/create_cifar10.sh to conver raw data to leveldb format.  
3. Run examples/cifar10/mpi_train_quick.sh to train the net. You can modify the   
"-n 16" to set new process number where 16 is the number of parallel processes,  
(if you use GPUs, the process number is m+1, m is GPU number)  
the "-host node11" is the node name in mpi_train_quick.sh script.  
### mnist  
1. Run data/mnist/get_mnist.sh to get mnist data.  
2. Run examples/mnist/create_mnist.sh to conver raw data to lmdb format.  
3. Run examples/mnist/mpi_train_lenet.sh to train the net. You can modify the 
"-n 16" to set new process number, the "-host node11" is the node name in 
mpi_train_quick.sh script.  
(if you use GPUs, the process number is m+1, m is GPU number)  

Change from BVLC/caffe
============================
1. framework
----------------------------
   a.used MPI to data-parallelism  
   b.each MPI process run one solve  
   c.training code is also mostly untouched  
   d.use a parameter server(thread),every solve compute each parameter , update to parameter server(PS) , PS compute and download new parameter to solve.  
2. class / files
----------------------------
   a.Solver/SGDSolver  
   b.data_layer/base_data_layer (parallel data read or distribute)  
   c.net (some interface and parameter update optimization)  
   d.other (include headfile, some interface, etc.)  
Acknowledgements
============================
The Caffe* parallel developers would like to thank  
QiHoo(Zhang,Gang ; Dr.Hu,Jinhui)  
Nvidia(Dr.Simon See ; Jessy Huan)  
for algorithm support and Inspur for guidance during Caffe* parallel development.  


