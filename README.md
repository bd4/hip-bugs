# Reproducers for performance issues and feature bugs in ROCm/HIP libraries

For batched_zgetrs.cxx, the input file is at
 https://www.mcs.anl.gov/~ballen/zgetrs.txt.bz2, and must be decompressed
with bunzip2 in the same directory as batched_zgetrs_{hip,nvcc}.
