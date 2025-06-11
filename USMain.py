import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant('cuda_ad_rgb')  # or 'cuda_ad_rgb' if you have CUDA

from CustomIntegrator import UltraIntegrator
mi.register_integrator("ultrasound_integrator", UltraIntegrator)

from CustomSensor import UltraSensor
mi.register_sensor("ultrasound_sensor", UltraSensor)

from CustomEmmitter import UltraRayEmitter
mi.register_emitter('ultrasound_emitter', UltraRayEmitter)

scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'ultrasound_integrator',
        'max_depth': 2
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
            'pixel_format': 'rgb',
            'component_format': 'float32'
        }
    },
    'emitter': {
        'type': 'ultrasound_emitter',
        'num_elements_lateral': 128,
        'radius': float('inf'),  # Linear array
        'plane_wave_angles_degrees': [-10, -5, 0, 5, 10],
        'center_frequency': 5e6,
        'to_world': mi.ScalarTransform4f().look_at(
            origin=[0, 0, 2],
            target=[0, 0, 0],
            up=[0, 1, 0]
        )
    },
    'shape': {
        'type': 'sphere',
        'center': [0, 0, 0],
        'radius': 1.0,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': [0.7, 0.1, 0.1]}
        }
    },
}

scene = mi.load_dict(scene_dict)
image = mi.render(scene)
real_part = image[:,:,0]  # All the result_real values arranged in 2D
imag_part = image[:,:,1]  # All the result_imag values arranged in 2D  
magnitude = image[:,:,2]  # All the magnitude values arranged in 2D

# Now magnitude will have your ultrasound data!
print("Magnitude range:", dr.min(magnitude), "to", dr.max(magnitude))

# Calculate magnitude properly
magnitude = dr.sqrt(real_part**2 + imag_part**2)
print("Magnitude range:", dr.min(magnitude), "to", dr.max(magnitude))

# Check for valid data
if dr.max(magnitude) > 0:
    # Apply ultrasound visualization
    magnitude_db = 20 * (dr.log(dr.maximum(magnitude, 1e-6)) / math.log(10))
    
    # Normalize to [0,1] for display
    display_image = (magnitude_db - dr.min(magnitude_db)) / (dr.max(magnitude_db) - dr.min(magnitude_db))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(display_image.numpy().clip(0, 1), cmap='gray')
    plt.title('Ultrasound Simulation')
    plt.axis('off')
    plt.show()
else:
    print("No ultrasound data found")