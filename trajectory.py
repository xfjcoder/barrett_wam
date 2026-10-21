"""A simple moving target: a horizontal circle around a (guaranteed-reachable) center point."""
import numpy as np


def circle_target(t, center, radius, omega):
    """Position on a horizontal (XY) circle at time t. omega in rad/s."""
    return center + radius * np.array([np.cos(omega * t), np.sin(omega * t), 0.0])
