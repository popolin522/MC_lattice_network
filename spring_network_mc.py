#!/usr/bin/env python3
"""
Authors: Po-An Lin @ Duke University @ Arya Lab
Contact: poan.lin@duke.edu
Monte Carlo simulation of 3D spring networks on FCC lattice. Implements with LAMMPS trajectory output.

"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import argparse
import freud


class SpringNetwork3D:
    
    def __init__(self, n, z=6.0, ksp=1.0, scale_lattice_constant=False, temperature=1.0, box_size=None):
        self.n = n
        self.z = z
        self.ksp = ksp
        self.temperature = temperature
        self.beta = 1.0 / temperature
        
        # Lattice setup
        if scale_lattice_constant:
            self.a0 = (3/self.ksp)**0.5  # scale with polymer end-to-end distance
        else:
            self.a0 = 1.0
            
        # Initialize system
        self.positions, self.N, self.L = self._create_fcc_lattice()
        print(f"Number of atoms: {self.N}")
        
        self.Nsp = int(z * self.N // 2)
        self.springs = self._create_spring_network()
        self.initial_positions = self.positions.copy()
        
        # Build adjacency for fast lookups
        self._build_adjacency()
        self.current_energy = self.calculate_energy()
        
        # Storage
        self.trajectory = []
        self.energies = []
        self.step_count = 0
        
    def _create_fcc_lattice(self):
        """Build FCC lattice using freud"""
        uc = freud.data.UnitCell.fcc()
        box, positions = uc.generate_system(self.n, scale=self.a0)
        
        positions = positions % box.Lx
        
        self.box = box
        return positions, len(positions), box.Lx
    
    def _create_spring_network(self):
        """Create spring network with target connectivity z"""
        springs = []
        
        # Find neighbors within cutoff
        cutoff = 1.1 * self.a0
        aabb = freud.locality.AABBQuery(self.box, self.positions)
        nlist = aabb.query(self.positions, dict(r_max=cutoff)).toNeighborList()
        
        # Build spring list (unique pairs only)
        for idx in range(len(nlist)):
            i = int(nlist.query_point_indices[idx])
            j = int(nlist.point_indices[idx])
            if i >= j:
                continue
                
            r_ij = self._distance_pbc(self.positions[i], self.positions[j])
            if r_ij < cutoff:
                springs.append((i, j, self.ksp, r_ij))
        
        # Randomly prune to target connectivity
        if len(springs) > self.Nsp:
            springs = random.sample(springs, self.Nsp)
        
        return springs
    
    def _distance_pbc(self, r1, r2):
        dr = r2 - r1
        dr = dr - self.L * np.round(dr / self.L)
        return np.linalg.norm(dr)
    
    def _vector_pbc(self, r1, r2):
        dr = r2 - r1
        dr = dr - self.L * np.round(dr / self.L)
        return dr
    
    def _build_adjacency(self):
        """Map each particle to its incident springs"""
        self.adjacency = [[] for _ in range(self.N)]
        for spring_idx, (i, j, _, _) in enumerate(self.springs):
            self.adjacency[i].append(spring_idx)
            self.adjacency[j].append(spring_idx)
    
    def calculate_energy(self, spring_indices=None):
        """Harmonic energy of network (or subset of springs)"""
        total_energy = 0.0
        
        if spring_indices is None:
            iterable = enumerate(self.springs)
        else:
            iterable = ((idx, self.springs[idx]) for idx in spring_indices)
        
        for _, (i, j, k_spring, r0) in iterable:
            if k_spring > 0:
                r = self._distance_pbc(self.positions[i], self.positions[j])
                total_energy += 0.5 * k_spring * (r - r0)**2
                
        return total_energy
    
    def calculate_forces(self):
        forces = np.zeros_like(self.positions)
        
        for i, j, k_spring, r0 in self.springs:
            if k_spring > 0:
                r_vec = self._vector_pbc(self.positions[j], self.positions[i])
                r = np.linalg.norm(r_vec)
                
                if r > 1e-10:
                    f_magnitude = k_spring * (r - r0)
                    f_vec = f_magnitude * r_vec / r
                    
                    forces[i] += f_vec
                    forces[j] -= f_vec
                    
        return forces
    
    def _local_energy_for_particle(self, particle):
        return self.calculate_energy(self.adjacency[particle])
    
    def mc_move(self, max_displacement=0.1):
        """Single MC move with Metropolis acceptance"""
        particle = random.randint(0, self.N - 1)
        
        old_pos = self.positions[particle].copy()
        old_local_energy = self._local_energy_for_particle(particle)
        
        # Trial move
        displacement = (np.random.random(3) - 0.5) * 2 * max_displacement
        new_pos = old_pos + displacement
        new_pos = new_pos % self.L
        self.positions[particle] = new_pos
        
        # Metropolis criterion
        new_local_energy = self._local_energy_for_particle(particle)
        delta_E = new_local_energy - old_local_energy
        
        if delta_E <= 0 or np.random.random() < np.exp(-self.beta * delta_E):
            self.current_energy += delta_E
            return True
        else:
            self.positions[particle] = old_pos
            return False
    
    def run_simulation(self, nsteps, sample_interval=100, max_displacement=0.1,
                      adaptive_step=True, target_acceptance=0.5, tune_interval=1000,
                      step_min=1e-4, step_max=1.0, adjust_factor=1.2):
        
        print(f"Starting MC simulation: {nsteps} steps")
        print(f"System: N={self.N}, z={self.z}, Nsp={len(self.springs)}")
        
        accepted_moves = 0
        recent_accepted = 0
        recent_total = 0
        ratio = 0
        current_step_size = max_displacement
        
        for step in range(nsteps):
            if self.mc_move(current_step_size):
                accepted_moves += 1
                recent_accepted += 1
            recent_total += 1
            
            # Adaptive step size tuning
            if adaptive_step and (step + 1) % tune_interval == 0:
                acc = recent_accepted / max(1, recent_total)
                if acc > 0:
                    ratio = acc / max(1e-6, target_acceptance)
                    ratio = min(max(ratio, 1/adjust_factor), adjust_factor)
                    current_step_size = current_step_size * ratio
                else:
                    current_step_size = current_step_size / adjust_factor
                
                current_step_size = min(max(current_step_size, step_min), step_max)
                recent_accepted = 0
                recent_total = 0
            
            # Sampling
            if step % sample_interval == 0:
                self.trajectory.append(self.positions.copy())
                self.energies.append(self.current_energy)
                self.step_count = step
                
                if step % (sample_interval * 10) == 0:
                    acceptance_rate = accepted_moves / (step + 1)
                    print(f"Step {step}, E={self.current_energy:.3f}, "
                          f"acc={acceptance_rate:.3f}, ratio={ratio:.3f}, step={current_step_size:.4e}")
        
        acceptance_rate = accepted_moves / nsteps
        print(f"Simulation complete. Final acceptance: {acceptance_rate:.3f}")
        
        return {
            'trajectory': self.trajectory,
            'energies': self.energies,
            'acceptance_rate': acceptance_rate
        }
    
    def calculate_msd(self):
        """MSD relative to initial ideal lattice positions"""
        if len(self.trajectory) < 1:
            return np.array([0]), np.array([0.0])
        
        trajectory = np.array(self.trajectory)
        nframes = len(trajectory)
        
        msd_values = []
        time_values = []
        
        for frame_idx in range(nframes):
            msd_sum = 0.0
            
            for particle in range(self.N):
                r_initial = self.initial_positions[particle]
                r_current = trajectory[frame_idx, particle]
                
                dr = r_current - r_initial
                dr = dr - self.L * np.round(dr / self.L)
                
                msd_sum += np.sum(dr**2)
            
            msd_values.append(msd_sum / self.N)
            time_values.append(frame_idx)
        
        return np.array(time_values), np.array(msd_values)
    
    def write_lammps_trajectory(self, filename, sample_stride=1):
        
        if not self.trajectory:
            print("No trajectory data available!")
            return
        
        print(f"Writing LAMMPS trajectory to {filename}...")
        
        with open(filename, 'w') as f:
            for frame_idx, positions in enumerate(self.trajectory[::sample_stride]):
                timestep = frame_idx * sample_stride
                
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{timestep}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{self.N}\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write(f"0.0 {self.L}\n")
                f.write(f"0.0 {self.L}\n") 
                f.write(f"0.0 {self.L}\n")
                f.write("ITEM: ATOMS id type x y z\n")
                
                for i, pos in enumerate(positions):
                    f.write(f"{i+1} 1 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        
        print(f"Trajectory written: {len(self.trajectory[::sample_stride])} frames")
    
    def write_lammps_bonds(self, filename):
        
        print(f"Writing LAMMPS bonds to {filename}...")
        
        with open(filename, 'w') as f:
            f.write("# LAMMPS data file for spring network\n\n")
            f.write(f"{self.N} atoms\n")
            f.write(f"{len(self.springs)} bonds\n")
            f.write("1 atom types\n")
            f.write("1 bond types\n\n")
            
            f.write(f"0.0 {self.L} xlo xhi\n")
            f.write(f"0.0 {self.L} ylo yhi\n")
            f.write(f"0.0 {self.L} zlo zhi\n\n")
            
            f.write("Masses\n\n")
            f.write("1 1.0\n\n")
            
            f.write("Atoms\n\n")
            for i, pos in enumerate(self.initial_positions):
                f.write(f"{i+1} 1 1 0.0 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            
            f.write("\nBonds\n\n")
            for bond_id, (i, j, k_spring, r0) in enumerate(self.springs):
                f.write(f"{bond_id+1} 1 {i+1} {j+1}\n")
        
        print(f"Bond data written: {len(self.springs)} bonds")


def run_connectivity_study(z, ksp, scale_lattice_constant):
    
    print("=== Connectivity Study ===")
    n = 10  # papers used 140x140 for 2D systems
    nsteps = 5000000
    
    os.makedirs('traj_data', exist_ok=True)
    os.makedirs('topo_data', exist_ok=True)
    
    print(f"\nRunning: z={z}, ksp={ksp}")
    
    network = SpringNetwork3D(n=n, z=z, temperature=1, ksp=ksp, 
                             scale_lattice_constant=scale_lattice_constant)
    sim_results = network.run_simulation(nsteps, sample_interval=500)
    
    times, msd = network.calculate_msd()
    
    results = {
        'network': network,
        'times': times,
        'msd': msd,
        'energies': sim_results['energies']
    }
    
    # Write outputs
    traj_file = f"traj_data/trajectory_z{z:.1f}_ksp{ksp:.1f}_scale{scale_lattice_constant}.lammpstrj"
    topo_file = f"topo_data/network_z{z:.1f}_ksp{ksp:.1f}_scale{scale_lattice_constant}.data"
    network.write_lammps_trajectory(traj_file, sample_stride=2)
    network.write_lammps_bonds(topo_file)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if len(results['times']) > 0:
        ax1.loglog(results['times'], results['msd'], 'o-', 
                  label=f'z={z}, ksp={ksp}', markersize=4)
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('MSD')
    ax1.set_title('Mean Square Displacement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(results['energies'], label=f'z={z}, ksp={ksp}')
    ax2.set_xlabel('MC Steps (×500)')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = f'spring_network_z{z:.1f}_ksp{ksp:.1f}_scale{scale_lattice_constant}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Spring Network Monte Carlo Simulation')
    parser.add_argument('--z', type=float, required=True, help='Connectivity parameter')
    parser.add_argument('--ksp', type=float, required=True, help='Spring constant')
    parser.add_argument('--scale_lattice_constant', action='store_true', 
                       help='Scale lattice constant with spring constant')
    args = parser.parse_args()
    
    random.seed(42)
    np.random.seed(42)
    
    print("3D Spring Network Monte Carlo Simulation")
    print("=" * 40)
    
    results = run_connectivity_study(args.z, args.ksp, args.scale_lattice_constant)
    
    print("\n=== Output Files ===")
    print(f"- traj_data/trajectory_z{args.z:.1f}_ksp{args.ksp:.1f}_scale{args.scale_lattice_constant}.lammpstrj")
    print(f"- topo_data/network_z{args.z:.1f}_ksp{args.ksp:.1f}_scale{args.scale_lattice_constant}.data")
    print(f"- spring_network_z{args.z:.1f}_ksp{args.ksp:.1f}_scale{args.scale_lattice_constant}.png")
    print("\nDone!")
