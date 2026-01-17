
# MjLab Microduck

## Training

```
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 2048 
```


With resume
```
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 2048 --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

## Play

```
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
```

## Imitation Learning

Train with imitation learning from reference motions:

```bash
# Setup: Link or copy your polynomial_coefficients.pkl file
ln -s ~/MISC/Open_Duck_reference_motion_generator/polynomial_coefficients.pkl \
      ~/MISC/mjlab_microduck/src/mjlab_microduck/data/reference_motions.pkl

# Train with imitation
uv run train Mjlab-Velocity-Flat-MicroDuck-Imitation --env.scene.num-envs 2048

# Play with imitation (phase observation included)
uv run play Mjlab-Velocity-Flat-MicroDuck-Imitation --wandb-run-path <...>
```

See [IMITATION_LEARNING.md](IMITATION_LEARNING.md) for detailed setup and configuration.