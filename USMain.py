import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('llvm_ad_rgb')  # or 'cuda_ad_rgb' if you have CUDA

from CustomIntegrator import UltraIntegrator
mi.register_integrator("ultrasound_integrator", UltraIntegrator)

from CustomSensor import UltraSensor
mi.register_sensor("ultrasound_sensor", UltraSensor)

from CustomEmmitter import UltraRayEmitter
mi.register_emitter('ultrasound_emitter', UltraRayEmitter)

scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'ultrasound_integrator',  # Standard Mitsuba path tracer
        'max_depth': 2
    },
    'sensor': {
        'type': 'ultrasound_sensor',
        'frequency': 5e6,
        'elements': 128,
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
        'type': 'ultra_emitter',
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
    }
}

scene = mi.load_dict(scene_dict)
image = mi.render(scene)
mi.Bitmap(image).write('basic_test.exr')

print("Rendered image shape:", image.shape)

if hasattr(image, 'numpy'):
    image_np = image.numpy()
else:
    # If image is a Bitmap, convert to numpy
    image_np = mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32).numpy()

# Clip values for display and show with matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(image_np.clip(0, 1))
plt.title('Rendered Mitsuba Image')
plt.axis('off')
plt.show()
