# `runtime/` — driver I/O scratch directory

`state_prop` reads `xd0.dat` (debris) and `xs0.dat` (spacecraft); each file
has seven lines in the order
`rx ry rz vx vy vz nominalTimeToTCA`, using km, km/s, and seconds (unscaled).
Outputs are written to `xdfKep.dat`, `xsfKep.dat`, `xdfJ2.dat`, `xsfJ2.dat`
(six lines each, same units as the inputs).

`state_back_prop` reads `xdTCA.dat` and `xsTCA.dat` (same 7-line format) and
writes `xd0Kep.dat`, `xs0Kep.dat`, `xd0J2.dat`, `xs0J2.dat`.

The `xd0.dat` / `xs0.dat` provided here describe an inclined (~51.6°),
slightly-eccentric LEO conjunction. **Avoid** equatorial (i≈0), polar
(i≈90°), or circular (e<1e-6) orbits — the mean↔osculating J2 transformation
has analytical singularities there (they exist in the original DACE port
too; see `hill2kep`'s `1/e` factor and `osculating2meanHill`'s
`sqrt(1-cos²i)` factor).

## Protocol

One forward propagation + TCA refinement:

```bash
./build/state_prop        # reads xd0.dat, xs0.dat; writes x{d,s}f{J2,Kep}.dat
```

One backward propagation from TCA to epoch0:

```bash
./build/state_back_prop   # reads x{d,s}TCA.dat; writes x{d,s}0{J2,Kep}.dat
```
