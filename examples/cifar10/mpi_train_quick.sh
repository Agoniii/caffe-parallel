#!/usr/bin/env sh

TOOLS=./build/tools

mpiexec.hydra -prepend-rank -host node11 -n 16 $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt

# reduce learning rate by factor of 10 after 8 epochs
#$TOOLS/caffe train \
#  --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
#  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate
