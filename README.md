# knn-bench

Evaluate, side by side, several kNN implementations.

# Usage

- build.sh: clean & rebuild the project.
- clean.sh: cleanup
- run.sh: run the benchmark

# Notes

The tests will run with the following arguments:

- Training points : 1024
- Testing points  : 64
- Dimension       : 3
- K               : 32

These arguments can be changed in `run.sh`.
