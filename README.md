# knn-bench

Evaluate, side by side, several kNN implementations.

# Usage

- `build.sh` : clean & rebuild the project.
- `clean.sh` : cleanup
- `run.sh`   : run the benchmark

# Notes

The tests will run with the following arguments:

- Training points : 1024
- Testing points  : 64
- Dimension       : 3
- K               : 32
- no validation   : -v

Currently the validation is done using a custom, exact, cpu implementation which is slow. Use `-v` to disable the validation when it takes too long.

These arguments can be changed in `run.sh`.
