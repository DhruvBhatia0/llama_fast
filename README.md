# llama_fast

## Installation instructions, tips

`git clone` the repository and change into the directory and then

```
git submodule update --init --recursive
```

## run base profiler

python3 -m torch.distributed.run --nproc_per_node=1 ../profile_base.py
