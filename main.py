
## main.py

from isingModel import IsingModel
import matplotlib.pyplot as plt

ising = IsingModel()
#ising.initialize_lattice("random") # To get a random lattice
ising.initialize_lattice(1) # To get a lattice with all 1's
#ising.initialize_lattice(-1) # To get a lattice with all -1's
#ising.initialize_lattice(0) # To get a lattice with all 0's

# information about the lattice
ising.print_lattice()
print("---- Creating the lattice ----")
print(f"Total Energy {ising.get_total_energy()}")
print(f"Total Magnetization {ising.magnetization()}")

# Run Monte Carlo simulation
print("---- Running Monte Carlo simulation ----")
ising.run_monte_carlo()
print("Simolation completed")
print(f"New total Energy {ising.get_total_energy()}")
print(f"New total Magnetization {ising.magnetization()}")
ising.print_lattice()

# Plot the energy and magnetization

plt.plot(ising.energy_history, label="Energy")
plt.plot(ising.magnetization_history, label="Magnetization")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.legend()
plt.show()