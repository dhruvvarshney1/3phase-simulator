import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

# Grid dimensions
Nx, Ny, Nz = 2, 2, 1
dx, dy, dz = 20.0, 20.0, 20.0  # ft

# Block indices
i_indices, j_indices, k_indices = np.meshgrid(
    np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij'
)

# Spatial coordinates (block centers)
X = (i_indices + 0.5) * dx
Y = (j_indices + 0.5) * dy
Z = (k_indices + 0.5) * dz

# Random porosity between 0.10 and 0.30
porosity_flat = np.random.uniform(0.10, 0.30, size=Nx * Ny * Nz)
porosity = porosity_flat.reshape(Nx, Ny, Nz)

# Create DataFrame
data = []
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            data.append({
                'i': i, 'j': j, 'k': k,
                'X': X[i, j, k], 'Y': Y[i, j, k], 'Z': Z[i, j, k],
                'porosity': porosity[i, j, k]
            })

df = pd.DataFrame(data)

# Print structured grid data
print("=== Synthetic 3D Reservoir Grid Data ===")
print(f"Grid dimensions: Nx={Nx}, Ny={Ny}, Nz={Nz}")
print(f"Block sizes: dx={dx} ft, dy={dy} ft, dz={dz} ft")
print()
print(df.to_string(index=False))
print()

# Porosity summary
print(f"Porosity range: {df['porosity'].min():.2f} - {df['porosity'].max():.2f}")
print(f"Mean porosity: {df['porosity'].mean():.4f}")
print()

# Display 2D porosity layout (top view, k=0)
porosity_2d = porosity[0, :, :]

print("=== 2D Porosity Layout (top view, k=0) ===")
print(porosity_2d)
print()

# Simple text-based visualization
print("=== Text-based 3D porosity visualization ===")
for k in range(Nz):
    print(f"Z = {Z[0, 0, k]:.1f} ft (layer {k}):")
    for j in range(Ny):
        row = ""
        for i in range(Nx):
            p = porosity[i, j, k]
            row += f"{p:.2f} "
        print(row)
    print()

# 3D visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of block centers colored by porosity
sc = ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), 
                c=porosity.ravel(), cmap='YlGn', 
                s=100, depthshade=True)

# Add text labels for each block
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            idx = i + j*Nx + k*Nx*Ny
            ax.text(X[i,j,k], Y[i,j,k], Z[i,j,k], 
                    f'({i},{j},{k})\nφ={porosity[i,j,k]:.2f}',
                    fontsize=8, ha='center')

# Set labels and title
ax.set_xlabel('X (ft)')
ax.set_ylabel('Y (ft)')
ax.set_zlabel('Z (ft)')
ax.set_title('3D Synthetic Reservoir Grid (2×2×1)')

# Set equal aspect ratio
max_range = max(X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min())/2.0
mid_x = (X.max()+X.min())/2.0
mid_y = (Y.max()+Y.min())/2.0
mid_z = (Z.max()+Z.min())/2.0
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Add colorbar
plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label='Porosity')

plt.tight_layout()
plt.show()

# 2D plot as well
plt.figure(figsize=(6, 5))
im = plt.imshow(porosity_2d, origin='lower', 
                extent=[0, dx*Ny, 0, dy*Nx], 
                cmap='YlGn', vmin=0.10, vmax=0.30)
plt.colorbar(im, label='Porosity')
plt.title('2D Porosity Layout (Top View)')
plt.xlabel('Y direction (ft)')
plt.ylabel('X direction (ft)')
plt.xlim(0, dx * Ny)
plt.ylim(0, dy * Nx)
plt.grid(True, alpha=0.3)
plt.show()