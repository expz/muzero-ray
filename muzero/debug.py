import numpy as np
import tensorflow_probability as tfp


def tensor_stats(t):
  return list(tfp.stats.quantiles(t, 4).numpy())