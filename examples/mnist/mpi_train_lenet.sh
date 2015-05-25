#!/usr/bin/env sh

mpiexec.hydra  -prepend-rank -host node11 -n 2 ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt
