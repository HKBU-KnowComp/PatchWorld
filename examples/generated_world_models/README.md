# Example Generated World Models

This folder contains the example induced world-model programs referenced in the
paper appendix section **"Induced world models by environment"**.

Included files:
- `alfworld_benchmark_world_model.py`
- `babyai_benchmark_world_model.py`
- `maze_benchmark_world_model.py`
- `sciworld_benchmark_world_model.py`
- `textcraft_benchmark_world_model.py`
- `webshop_benchmark_world_model.py`
- `wordle_benchmark_world_model.py`

These examples were copied from the paper's `generated_world_models/` appendix
artifacts and adapted for this standalone repo by switching:

```python
from abductworld.worldmodel_base import BaseWorldModel
```

to:

```python
from patchworld.worldmodel_base import BaseWorldModel
```
