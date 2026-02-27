
# MjLab Microduck

## Training

```
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```


With resume
```
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

## Play

```
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
```

## ONNX export 

```
uv run export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
````