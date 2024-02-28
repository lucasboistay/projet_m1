## main.py

from isingModel import IsingModel
import matplotlib.pyplot as plt
import time
import numpy as np

print("---- Creating the lattice ----")
ising = IsingModel(50, 50, 300, 1000)
# ising.initialize_lattice("random") # To get a random lattice
ising.initialize_lattice(1) # To get a lattice with all 1's
# ising.initialize_lattice(-1)  # To get a lattice with all -1's
# ising.initialize_lattice(0) # To get a lattice with all 0's

# information about the lattice
starting_lattice = np.copy(ising.get_lattice())

print(f"Total Energy {ising.get_total_energy()}")
print(f"Total Magnetization {ising.magnetization()}")

# Run Monte Carlo simulation
print("---- Running Monte Carlo simulation ----")
start = time.time()
ising.run_monte_carlo()
end = time.time()
print("Simolation completed")
print(f"New total Energy {ising.get_total_energy()}")
print(f"New total Magnetization {ising.magnetization()}")
print(f"Time taken {end - start} seconds")
ending_lattice = np.copy(ising.get_lattice())

# Plot the energy and magnetization

plt.plot(ising.energy_history, label="Energy")
plt.plot(ising.magnetization_history, label="Magnetization")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.legend()
plt.savefig("data/energy_magnetization.png")
plt.show()

# Plot the lattice with heat map on the same figure

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(starting_lattice, cmap="hot")
plt.title("Initial lattice")

plt.subplot(1, 2, 2)
plt.imshow(ending_lattice, cmap="hot")
plt.title("Final lattice")
plt.show()

plt.savefig("data/lattice.png")

