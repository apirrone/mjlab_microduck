
# MjLab Microduck

Main repo : https://github.com/apirrone/microduck

https://github.com/user-attachments/assets/e9a0d4de-8a3d-4e44-b490-e873728cf2bf


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
