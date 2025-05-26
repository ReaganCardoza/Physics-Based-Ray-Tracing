'''This was working initially in 2D but got lost in 3D and lacked control over
Transducer location'''
import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
from mitsuba.scalar_rgb import ScalarPoint3f
import random


mi.set_variant("cuda_ad_rgb")

scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'path'
    },
    'cylinder': {
        'type': 'cylinder',
        'radius': 0.2,
        'p0': [0, -0.5, 0],
        'p1': [0, 0.5, 0],
        'bsdf': {
            'type': 'roughconductor',
            'alpha': 0.1,  # low roughness = strong specular reflection
            'distribution': 'ggx'
        }
    },
    'sensor': {
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f().look_at(
            origin=ScalarPoint3f(0, 0, 2),        # simulate transducer 2 units in front
            target=ScalarPoint3f(0, 0, 0),        # looking at the cylinder
            up=ScalarPoint3f(0, 1, 0)
        ),
        'film': {
            'type': 'hdrfilm',
            'width': 64,
            'height': 64,
            'rfilter': { 'type': 'box' }
        },
        'sampler': {
            'type': 'independent',
            'sample_count': 16
        }
    }
}

scene = mi.load_dict(scene_dict)

#Ray gen and transducer setup
num_elements = 10
rays_per_element = 10
angle_range_deg = 10



#position of transducer evenly spaced out
x_positions = np.linspace(-0.3, 0.3, num_elements)
y_positions = np.linspace(-0.3, 0.3, 10)  # probe sweep vertically
origin_z = 2.0 #Set equal to the transducer positon of it changes


# GENERATING PRIMARY RAYS

intersections = []

for y in y_positions:
    for x in x_positions:
        for _ in range(rays_per_element):

            angle_deg = np.random.uniform(-angle_range_deg, angle_range_deg)
            angle_rad = np.radians(angle_deg)

            dx = float(np.sin(angle_rad))  # fan in X
            dz = float(-np.cos(angle_rad))  # into the scene
            dy = float(np.random.uniform(-0.01, 0.01))  # optional small spread in Y

            origin = mi.Point3f(float(x), float(y), float(origin_z))
            direction = mi.Vector3f(float(dx), float(dy), float(dz))

            ray = mi.Ray3f(o=origin, d=dr.normalize(direction))
            si = scene.ray_intersect(ray)

            if bool(si.is_valid().numpy()[0]):
                hit_x = float(si.p.x.numpy()[0])
                hit_y = float(si.p.y.numpy()[0])
                hit_z = float(si.p.z.numpy()[0])
                intersections.append((hit_x, hit_y, hit_z))

# Print some results
print(f"Total rays traced: {num_elements * rays_per_element}")
print(f"Number of intersections: {len(intersections)}")
for hit in intersections[:5]:
    print(f"Hit at x = {hit[0]:.3f}, z = {hit[1]:.3f}")


#Primary Ray plotting begining ----------------------------------
# Unpack intersection points
hit_xs = [x for x, y, z in intersections]
hit_zs = [z for x, y, z in intersections]
hit_ys = [y for x, y, z in intersections]


# Plot hit locations
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(hit_xs, hit_ys, hit_zs, color='blue', label='Hit Points')

# Cylinder mesh (aligned vertically along z-axis)
cylinder_radius = 0.2
cylinder_height = 1.0
z_min = 0.0
z_max = 1

# Create cylinder surface mesh
y = np.linspace(-0.5, 0.5, 50)  # cylinder height in Y
theta = np.linspace(0, 2 * np.pi, 50)
theta_grid, y_grid = np.meshgrid(theta, y)

x_grid = cylinder_radius * np.cos(theta_grid)
z_grid = cylinder_radius * np.sin(theta_grid)

# Plot cylinder as semi-transparent surface
ax.plot_surface(
    x_grid, y_grid, z_grid,
    alpha=0.05, color='gray', edgecolor='none'
)

ax.set_title("3D Primary Ray Hits on Cylinder")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()
#------------ Primary Ray Plotting END ------------------


#This section is no longer working because it is not adapted for 3D


#GENERATING SECONDAY RAYS
# Secondary ray tracing from hit points back to transducer
echo_data = {}  # per transducer element
speed_of_sound = 1.54  # mm/us
cutoff_angle = 30  # degrees

for hit_x, hit_z in intersections:
    # Choose a random transducer element
    element_idx = random.randint(0, num_elements - 1)
    tx = x_positions[element_idx]
    transducer_pt = mi.Point3f(float(tx), 0, float(origin_z))
    hit_pt = mi.Point3f(hit_x, 0, hit_z)

    # Create secondary ray
    direction = dr.normalize(transducer_pt - hit_pt)
    ray = mi.Ray3f(o=hit_pt, d=direction)
    si2 = scene.ray_intersect(ray)

    if not bool(si2.is_valid().numpy()[0]):
        # No obstruction; echo returns
        distance = dr.norm(transducer_pt - hit_pt)
        time = (distance / speed_of_sound).numpy()[0]

        # Compute angle between incoming ray and surface normal
        angle = dr.acos(dr.dot(-direction, mi.Vector3f(0, 0, -1)))
        angle_deg = np.degrees(angle.numpy()[0])


        # Weighting based on directivity falloff
        if angle_deg <= cutoff_angle:
            weight = (cutoff_angle - angle_deg) / cutoff_angle
            echo_data[element_idx].append((time, weight))

for i in range(num_elements):  # First 3 elements
    print(f"\nElement {i}:")
    for t, w in echo_data[i][:5]:
        print(f"  Time = {t:.2f} µs, Weight = {w:.2f}")


#Pressure-Time Signal fro one Transducer

# Simulation Parameters
fc = 2e6              # 2 MHz center frequency
sigma = 0.1           # Gaussian envelope width (µs)
sampling_rate = 5e6   # 5 MHz sampling rate
duration = 3.0        # Total duration (µs)
num_samples = int(duration * sampling_rate)
t = np.linspace(0, duration, num_samples)

# Initialize signal matrix [num_elements x num_samples]
pressure_signals = np.zeros((num_elements, num_samples))

# Pulse model (UltraRay Eq. 14)
def pulse(t, t0, amp, fc, sigma):
    carrier = np.sin(2 * np.pi * fc * (t - t0))
    envelope = np.exp(-((t - t0)**2) / sigma**2)
    return amp * carrier * envelope

# Fill in pressure signal for each transducer element
for element_idx in range(num_elements):
    for time_us, weight in echo_data[element_idx]:
        pressure_signals[element_idx] += pulse(t, time_us, weight, fc, sigma)

#visualize a few rows
plt.figure(figsize=(12, 6))
for i in range(min(5, num_elements)):
    plt.plot(t, pressure_signals[i], label=f"Element {i}")
plt.title("Pressure-Time Signals for Multiple Transducer Elements")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
