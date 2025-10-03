# NCT Protonated Methane: Neural Canonical Transformation calculation of Vibrational Energy Levels of Protonated Methane CH5+

This repo is implemented using [jax](https://github.com/jax-ml/jax)

Usage:

```bash
python3 -m neuralvib.train \
    --folder "/home/CH5/JBB/"  \
    --molecule "CH5+"\
    --select_potential "J.Phys.Chem.A2006,110,1569-1574"\
    --num_of_particles=6\
    --num_orb=136  \
    --flow_type "RNVP"  \
    --flow_depth=32\
    --mlp_width=64 \
    --mlp_depth=2\
    --batch=100 \
    --acc_steps=1\
    --epoch=300000  \
    --clip_factor=5\
    --optimizer "adam"\
    --adam_lr=1e-4  \
    --mc_therm=20 \
    --mc_steps=100  \
    --mc_stddev=2.0 \
    --mc_selfadjust_stepsize \
    --excite_gen_type=3 \
    --pretrain-network \
    --pretrain-batch=100000 \
    --pretrain-epochs=100 \
```