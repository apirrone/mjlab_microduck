
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