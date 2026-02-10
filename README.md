# MC_lattice_network
Monte Carlo simulation of thermalized spring (fcc) networks. Runs MC moves with Metropolis acceptance and dumps trajectories for visualization in OVITO.

## Quick start

```bash
python spring_network_mc.py --z 6.0 --ksp 1.0
```

Options:
- `--z`: connectivity (average number of springs per node)
- `--ksp`: spring constant
- `--scale_lattice_constant`: scale lattice spacing with spring constant. Not useful.

## Output

- `traj_data/`: LAMMPS trajectory files
- `topo_data/`: bond topology files
- PNG plots of MSD and energy vs time (check equil.)

## Notes

System size is hardcoded at n=10 (gives ~1000 atoms for FCC). Edit `run_connectivity_study()` if you want bigger systems.

Needs `freud` for lattice generation. It's really straightforward to switch to a different lattice.

For more information on this topic, read papers by Fred MacKintosh(10.1103/PhysRevLett.111.095503), Xiaoming Mao (10.1103/PhysRevE.93.022110), or Michael Thorpe (10.1103/PhysRevE.76.041135). Those papers contain the real insights; this repo is merely an exploration of related ideas in the context of polymer-integrated protein crystals.
