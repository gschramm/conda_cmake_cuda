# Minimal example to investigate conda build + cmake + cuda

When running conda build, you have to specify the nvidia channel
which provides the cuda-toolkit
```
conda build . -c nvidia
```
