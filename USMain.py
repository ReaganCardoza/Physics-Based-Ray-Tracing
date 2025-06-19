import mitsuba as mi
import drjit as dr
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import hilbert


mi.set_variant("cuda_ad_mono")

from CustomIntegrator import UltraIntegrator
mi.register_integrator("ultrasound_integrator", UltraIntegrator)

from CustomSensor import UltraSensor
mi.register_sensor("ultrasound_sensor", UltraSensor)

from CustomEmmitter import UltraRayEmitter
mi.register_emitter('ultrasound_emitter', UltraRayEmitter)

from CustomBSDF import UltraBSDF
mi.register_bsdf('ultrasound_bsdf', UltraBSDF)

scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'ultrasound_integrator',
        'max_depth': 3
    },
        'sensor': {
        'type': 'ultrasound_sensor',
        'num_elements_lateral': 128,
        'elements_width': 0.003,
        'elements_height': 0.01,
        'pitch': 0.00035,
        'radius': float('inf'),  # Linear array
        'center_frequency': 5e6,
        'sound_speed': 1540,
        'directivity': 1.0,
        'to_world': mi.ScalarTransform4f().look_at(
            origin=[0, 0, 2],
            target=[0, 0, 0],
            up=[0, 1, 0]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 512,
            'pixel_format': 'luminance',
            'component_format': 'float32'
        }
    },
    'emitter': {
        'type': 'ultrasound_emitter',
        'num_elements_lateral': 128,
        'radius': float('inf'),  # Linear array
        'plane_wave_angles_degrees': list(range(-30, 35, 5)),
        'center_frequency': 5e6,
        'to_world': mi.ScalarTransform4f().look_at(
            origin=[0, 0, 2],
            target=[0, 0, 0],
            up=[0, 1, 0]
        )
    },
    'shape': {
        'type': 'sphere',
        'center': [0, 0, 1.95],
        'radius': 0.01,
        'bsdf': {
            'type': 'ultrasound_bsdf',
            'impedance': 7.8,
            'roughness': 0.5
        }
    },
}

scene = mi.load_dict(scene_dict)


# Render
image = mi.render(scene)

# Retrive channel data from integrator
integrator = scene.integrator()
channel_buf = integrator.channel_buf.numpy()
n_angles = integrator.n_angles
n_elements = integrator.n_elements
time_samples = integrator.time_samples

print("channel_buf sum:", np.sum(channel_buf))
print("channel_buf max:", np.max(channel_buf))


# Reshape to 3D tensor
channel_data = channel_buf.reshape((n_angles, n_elements, time_samples))

# Axial pulse convolution
fs = integrator.fs
fc = integrator.frequency
sigma = integrator.wave_cycles / (2 * np.pi * fc)
t = np.arange(time_samples) / fs
pulse = np.sin(2 * np.pi * fc * t) * np.exp(-t**2 / sigma)

convolved = np.zeros_like(channel_data)
for a in range(n_angles):
    for e in range(n_elements):
        convolved[a, e] = np.convolve(channel_data[a, e], pulse, mode='same')

# Delay and sum beamforming
# After beamforming
bmode = np.zeros((time_samples, n_angles))  # 2D array: depth x angles

for a_idx in range(n_angles):
    # Process each angle
    rf_line = np.sum(convolved[a_idx], axis=0)
    rf_line -= np.mean(rf_line)
    analytic_signal = hilbert(rf_line)
    envelope = np.abs(analytic_signal)
    bmode[:, a_idx] = envelope  # Fill column in 2D array

# Log compression
# 1. Skip normalization - use raw envelope values
bmode_db = 20 * np.log10(bmode + 1e-12)  # Absolute dB values

dynamic_range = 60
max_db = np.max(bmode_db)
min_db = max_db - dynamic_range
bmode_db_clipped = np.clip(bmode_db, min_db, max_db)
display_image = (bmode_db_clipped - min_db) / dynamic_range

depth_axis_mm = (np.arange(time_samples) / fs) * (1540 / 2) * 1e3  # mm

plt.figure(figsize=(10, 8))
extent = [0, n_angles, depth_axis_mm[-1], depth_axis_mm[0]]
im = plt.imshow(display_image.T, aspect='auto', cmap='gray', extent=extent, origin='upper')
plt.xlabel('Scan Line Index')
plt.ylabel('Depth (mm)')
plt.title('Simulated Ultrasound B-mode Image (UltraRay)')
plt.colorbar(im, label='Relative Echo Intensity (dB)')
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Debugging
print("Raw envelope min:", np.min(bmode), "max:", np.max(bmode))
print("Log values min:", np.min(bmode_db), "max:", np.max(bmode_db))
print("After dynamic range min:", np.min(bmode_db_clipped), "max:", np.max(bmode_db_clipped))




