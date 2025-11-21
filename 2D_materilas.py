"""
@author: Ehsanab

This program create random defect in 6 TMDs, phosphorene and graphene
The user can change the size of structures based on experimental STM or ... images

"""
import pybinding as pb
import matplotlib.pyplot as plt
from pybinding.repository import group6_tmd, graphene, phosphorene
import random
import numpy as np
import os

# Create the directory if it doesn't exist
output_path = r"path to destination folder to save"
os.makedirs(output_path, exist_ok=True)

def vacancy(position, radius):
    @pb.site_state_modifier
    def modifier(state, x, y):
        x0, y0 = position
        state[(x-x0)**2 + (y-y0)**2 < radius**2] = False
        return state
    return modifier

def generate_random_defects(num_defects=3, x_range=(-2, 2), y_range=(-2, 2), radius_range=(0.1, 0.4)):
    """Generate random defect positions and radii"""
    defects = []
    for _ in range(num_defects):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        radius = random.uniform(radius_range[0], radius_range[1])
        defects.append((x, y, radius))
    return defects

# Define materials with explicit sublattice colors
materials_config = {
    "MoS2": {
        "model": group6_tmd.monolayer_3band("MoS2"),
        "colors": ["lightblue", "pink"]  # Mo, S
    },
    "WS2": {
        "model": group6_tmd.monolayer_3band("WS2"), 
        "colors": ["blue", "darkred"]  # W, S
    },
    "MoSe2": {
        "model": group6_tmd.monolayer_3band("MoSe2"),
        "colors": ["lightblue", "orange"]  # Mo, Se
    },
    "WSe2": {
        "model": group6_tmd.monolayer_3band("WSe2"),
        "colors": ["blue", "yellow"]  # W, Se
    },
    "MoTe2": {
        "model": group6_tmd.monolayer_3band("MoTe2"),
        "colors": ["lightblue", "green"]  # Mo, Te
    },
    "WTe2": {
        "model": group6_tmd.monolayer_3band("WTe2"),
        "colors": ["blue", "darkgreen"]  # W, Te
    },
    "phosphorene": {
        "model": phosphorene.monolayer_4band(),
        "colors": ["purple", "magenta", "violet", "lavender"]  # Multiple P atoms
    },
    "graphene": {
        "model": graphene.monolayer(),
        "colors": ["gray", "black"]  # Two carbon sublattices
    }
}

# Set random seed for reproducibility
random.seed(42)

# Create a text file to save all defect information
info_file = open(os.path.join(output_path, "defect_information.txt"), "w")
info_file.write("DEFECT INFORMATION FOR 2D MATERIALS\n")
info_file.write("=" * 60 + "\n\n")

print("="*60)
print("Plotting all materials with random defects:")
print("="*60)

# Simple and reliable plotting function
def plot_material_simple(name, material_config, defects, filename_suffix=""):
    """Simple plotting using PyBinding's built-in plot function"""
    
    # Create model
    model = pb.Model(
        material_config["model"],
        pb.rectangle(x=5, y=5),
        *[vacancy(position=[x, y], radius=r) for x, y, r in defects]
    )
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Use PyBinding's plot with the color list
    model.plot(
        site={
            'radius': 0.1, 
            'cmap': material_config["colors"]
        }
    )
    
    plt.title(f"{name} with {len(defects)} Defects", fontsize=16, fontweight='bold')
    plt.gca().set_frame_on(False)
    plt.axis('off')
    
    # Save plot
    filename = f"{name}{filename_suffix}.png"
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return len(defects)

# Alternative: Manual plotting with correct position access
def plot_material_manual(name, material_config, defects, filename_suffix=""):
    """Manual plotting with correct position handling"""
    
    # Create model
    model = pb.Model(
        material_config["model"],
        pb.rectangle(x=5, y=5),
        *[vacancy(position=[x, y], radius=r) for x, y, r in defects]
    )
    
    # Get system data - this is the correct way to access positions
    system = model.system
    x_pos = system.positions[0]  # x coordinates
    y_pos = system.positions[1]  # y coordinates
    sublattices = system.sublattices
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Get unique sublattices
    unique_sublattices = np.unique(sublattices)
    
    # Plot each sublattice with its assigned color
    for i, sub_idx in enumerate(unique_sublattices):
        mask = sublattices == sub_idx
        color_idx = i % len(material_config["colors"])  # Ensure we don't go out of bounds
        
        plt.scatter(x_pos[mask], y_pos[mask], 
                   c=material_config["colors"][color_idx], 
                   s=200, label=f'Sublattice {sub_idx}', alpha=0.8)
    
    plt.title(f"{name} with {len(defects)} Defects (Manual)", fontsize=16, fontweight='bold')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.gca().set_frame_on(False)
    plt.axis('off')
    
    # Save plot
    filename = f"{name}{filename_suffix}_manual.png"
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# Plot all materials using the simple method
info_file.write("ALL MATERIALS WITH RANDOM DEFECTS:\n")
info_file.write("=" * 40 + "\n")

for name, config in materials_config.items():
    # Generate random defects
    random_defects = generate_random_defects(num_defects=random.randint(2, 4))
    
    # Method 1: Simple PyBinding plot
    num_defects = plot_material_simple(name, config, random_defects)
    
    # Method 2: Manual plot (optional)
    plot_material_manual(name, config, random_defects)
    
    # Save defect information
    defect_info = f"{name}: {len(random_defects)} defects at positions {[(round(x, 2), round(y, 2), round(r, 2)) for x, y, r in random_defects]}"
    print(defect_info)
    info_file.write(defect_info + "\n")

# Create combined figure with all materials
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()

for i, (name, config) in enumerate(materials_config.items()):
    # Generate new random defects for combined plot
    random_defects = generate_random_defects(num_defects=random.randint(2, 3))
    
    # Create model
    model = pb.Model(
        config["model"],
        pb.rectangle(x=5, y=5),
        *[vacancy(position=[x, y], radius=r) for x, y, r in random_defects]
    )
    
    # Plot using PyBinding's simple plot method on subplot
    plt.sca(axes[i])
    model.plot(site={'radius': 0.08, 'cmap': config["colors"]})
    axes[i].set_title(f"{name}", fontsize=12, fontweight='bold')
    axes[i].set_frame_on(False)
    axes[i].axis('off')

plt.tight_layout()
combined_filename = "all_materials_combined.png"
combined_filepath = os.path.join(output_path, combined_filename)
plt.savefig(combined_filepath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Create TMD-specific comparison
tmd_materials = {name: config for name, config in materials_config.items() 
                if name in ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, config) in enumerate(tmd_materials.items()):
    random_defects = generate_random_defects(num_defects=3)
    
    model = pb.Model(
        config["model"],
        pb.rectangle(x=5, y=5),
        *[vacancy(position=[x, y], radius=r) for x, y, r in random_defects]
    )
    
    plt.sca(axes[i])
    model.plot(site={'radius': 0.1, 'cmap': config["colors"]})
    axes[i].set_title(f"{name}", fontsize=14, fontweight='bold')
    axes[i].set_frame_on(False)
    axes[i].axis('off')

plt.tight_layout()
tmd_filename = "TMD_comparison.png"
tmd_filepath = os.path.join(output_path, tmd_filename)
plt.savefig(tmd_filepath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Create detailed individual plots
print("\n" + "="*60)
print("Creating detailed individual plots:")
print("="*60)
info_file.write("\nDETAILED INDIVIDUAL PLOTS:\n")
info_file.write("=" * 30 + "\n")

for name, config in materials_config.items():
    # Generate new random defects for detailed plots
    random_defects = generate_random_defects(num_defects=random.randint(3, 5))
    
    # Create detailed plot with larger size
    plt.figure(figsize=(12, 10))
    
    model = pb.Model(
        config["model"],
        pb.rectangle(x=6, y=6),
        *[vacancy(position=[x, y], radius=r) for x, y, r in random_defects]
    )
    
    model.plot(site={'radius': 0.12, 'cmap': config["colors"]})
    plt.title(f"{name} - Detailed View", fontsize=18, fontweight='bold')
    plt.gca().set_frame_on(False)
    plt.axis('off')
    
    # Save detailed plot
    detailed_filename = f"{name}_detailed.png"
    detailed_filepath = os.path.join(output_path, detailed_filename)
    plt.savefig(detailed_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Print and save detailed defect information
    detailed_info = f"{name}: {len(random_defects)} random defects"
    print(f"\n{detailed_info}")
    info_file.write(detailed_info + "\n")
    for j, (x, y, r) in enumerate(random_defects):
        defect_detail = f"  Defect {j+1}: position=({x:.2f}, {y:.2f}), radius={r:.2f}"
        print(defect_detail)
        info_file.write(defect_detail + "\n")

# Close the info file
info_file.close()

# Print summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"All plots and information saved to: {output_path}")
print("\nFiles created for each material:")
print("- [material_name].png (simple PyBinding plot)")
print("- [material_name]_manual.png (manual scatter plot)") 
print("- [material_name]_detailed.png (detailed view)")
print("- all_materials_combined.png (all in one figure)")
print("- TMD_comparison.png (TMD materials only)")
print("- defect_information.txt (defect coordinates)")

print(f"\nColor schemes used:")
for material, config in materials_config.items():
    print(f"- {material}: {config['colors']}")
