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

To run a script `main.py` (for example some figures in `./tex/figures/` are
generated from scripts):

```
$ uv run main.py
```

To run a notebook server with the environment:

```
$ uv run jupyter notebook
```
