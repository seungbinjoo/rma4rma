# Policy Training:

To execute policy training

```bash
python main.py --phase="PolicyTraining" --env_id="PickSingleYCB-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=1_000_000 --eval_freq=10 --num-steps=20
```