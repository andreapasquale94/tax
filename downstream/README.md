# Downstream projects

This directory contains downstream consumers of the `tax` library. They are
not part of the core tax build and are not included when `tax` is used as a
dependency elsewhere.

| Project | Description |
|---------|-------------|
| [`cam/`](cam/) | C++23 port of [arma1978/multi-impulsive-cam](https://github.com/arma1978/multi-impulsive-cam) — collision-avoidance maneuver propagation using tax instead of DACE. |
