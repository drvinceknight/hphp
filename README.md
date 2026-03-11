# Hurt People Hurt People

A research paper on intergenerational trauma

## Software

The code here is packaged using [`uv`](https://docs.astral.sh/uv/). After
installing `uv` run:

```
$ uv sync
```

To run the tests:

```
$ uv run pytest
```

To lint with ruff:

```
$ uv run ruff check src/
```

To type check with ty:

```
$ uv run ty check src/
```

To run a script `main.py` (for example some figures in `./tex/figures/` are
generated from scripts):

```
$ uv run main.py
```

To run a notebook server with the environment:

```
$ uv run jupyter notebook
```

## Simulation data and pipeline (DVC)

The raw simulation data (~5 GB) is tracked with [DVC](https://dvc.org) rather
than git. The pipeline (`dvc.yaml`) reproduces the figures and paper PDF from
that data.

### Fetching data and reproducing outputs (local machine)

```
$ rsync -av harpy:hphp/tex/figures/figure_4/data/raw/ tex/figures/figure_4/data/raw/
$ uv run dvc repro         # rebuilds figures and paper (only changed stages)
```

### After new simulations (on the server)

```
$ uv run dvc add tex/figures/figure_4/data/raw   # update the data manifest
$ git add tex/figures/figure_4/data/raw.dvc
$ git commit -m "Update simulation data"
$ git push
```
