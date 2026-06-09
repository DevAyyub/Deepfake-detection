"""sfdet.utils — shared infrastructure.

  config.py    Load and layer YAML configs (base <- data <- model <- experiment)
               into one resolved config object.
  paths.py     Read paths.yaml (gitignored) and resolve dataset/crop roots, so no
               absolute path enters version control or the tracked configs.
  seed.py      Seed Python/NumPy/torch and set deterministic flags for
               reproducible runs.
  registry.py  Name -> class registries (models, datasets, losses) so a config
               string selects an implementation. This is the mechanism that
               makes ablations config-only: a model variant is just a registry
               key.
  logging.py   Console + TensorBoard logging setup (run dirs are gitignored).

Implementations land alongside the first training/eval code.
"""
