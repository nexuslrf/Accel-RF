# NeRF Example

This is a minimal implementation of orginal NeRF, for the Test purpose.

This implementatation is 10% faster than [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)

**TODO** more options will be gradually added. 🙂

To run this example: 
```bash
python example/nerf/run_nerf.py --config example/configs/lego.txt
```
or 
```bash
python -m torch.distributed.launch --nproc_per_node=4 example/nerf/run_nerf.py --config example/configs/lego.txt
```