# Policy Training:

To execute policy training

```bash
python base_policy.py --exp_name="BasePolicy_PickYCBSingleRMA_DDMMYY" --phase="PolicyTraining" --env_id="PickSingleYCBRMA-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=50_000_000 --eval_freq=25 --num-steps=50
```

To execute adaptation training

```bash
python adaptation.py --exp_name="AdaptationTraining_PickYCBSingleRMA_DDMMYY" --phase="AdaptationTraining" --env_id="PickSingleYCBRMA-v1" \
  --num_envs=256 --base_policy_checkpoint="/users/joo/4yp/rma4rma/runs/BasePolicy_PickYCBSingleRMA_09012025/final_ckpt.pt"
```