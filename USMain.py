import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt


# Import Ultraspy modules
from ultraspy.beamformers.das import DelayAndSum
from ultraspy.scan import GridScan
from ultraspy.probes.factory import build_probe

mi.set_variant("llvm_ad_mono")

from CustomIntegrator import UltraIntegrator
mi.register_integrator("ultrasound_integrator", UltraIntegrator)

from CustomSensor import UltraSensor
mi.register_sensor("ultrasound_sensor", UltraSensor)

from CustomEmmitter import CustomEmitter
mi.register_emitter('ultrasound_emitter', CustomEmitter)

from CustomBSDF import UltraBSDF
mi.register_bsdf('ultrasound_bsdf', UltraBSDF)

def us_render():
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'ultrasound_integrator',
            'max_depth': 100, # Keep at 1 for simpler debugging
            'sampling_rate': 50e6,
            'frequency': 5e6,
            'sound_speed': 1540,
            'attenuation': 0.5, # Keep at 0.0 for debugging signal strength
            'wave_cycles': 5,
            'main_beam_angle': 24,
            'cutoff_angle': 30,
            'n_elements': 64, # Keep low for faster debugging
            'pitch': 0.00035,
            'time_samples': 10000, # Keep large enough
            'angles': dr.linspace(mi.Float, -10, 10, 5) # Keep low for faster debugging
        },
            'sensor': {
            'type': 'ultrasound_sensor',
            'num_elements_lateral': 1280,
            'elements_width': 0.003,
            'elements_height': 0.01,
            'pitch': 0.00035,
            'radius': float('inf'),  # Linear array
            'center_frequency': 5e6,
            'sound_speed': 1540,
            'directivity': 1.0,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, 0.0],
                target=[0, 0, 0.03],
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
       'shape': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate(mi.ScalarVector3f(0.0, 0.0, 0.01)).scale(mi.ScalarPoint3f(0.01, 0.01, 0.01)),
            'bsdf': {
                'type': 'ultrasound_bsdf',
                'impedance': 7.8,
                'roughness': 0.1
            }
        },
    }

    scene = mi.load_dict(scene_dict)

    # Get the integrator instance
    integrator = scene.integrator()

    # Call the custom simulation method on the integrator
    print("Starting Mitsuba custom acquisition simulation...")
    integrator.simulate_acquisition(scene) # This replaces mi.render(scene)
    print("Mitsuba custom acquisition simulation finished.")

    # Retrieve data after simulation
    channel_buf = integrator.channel_buf.numpy()
    transmission_delays_flat = integrator.transmission_delays_buf.numpy()

    n_angles = integrator.n_angles
    n_elements = integrator.n_elements
    time_samples = integrator.time_samples
    fs = integrator.fs
    fc = integrator.frequency
    c = integrator.sound_speed
    angles_deg = integrator.angles.numpy()

    print("channel_buf sum:", np.sum(channel_buf))
    print("channel_buf max:", np.max(channel_buf))


    # Reshape to 3D tensor (n_angles, n_elements, time_samples)
    channel_data = channel_buf.reshape((n_angles, n_elements, time_samples))

    # Reshape delays to (n_angles, n_elements)
    transmission_delays = transmission_delays_flat.reshape((n_angles, n_elements))


#Ultraspy Integration Starts Here

    #Prepare Probe Information
    probe = build_probe(
        geometry_type='linear',
        nb_elements=n_elements,
        pitch=integrator.pitch,
        central_freq=fc,
        bandwidth=70
    )

    #Use the captured transmission delays directly
    ul_delays = transmission_delays

    #Define Sequence Elements
    elements_indices = np.arange(n_elements)
    sequence = {
        'emitted': np.tile(elements_indices, (n_angles, 1)),
        'received': np.tile(elements_indices, (n_angles, 1)),
    }

    #Create acquisition_info and data_info dictionaries
    acquisition_info = {
        'sampling_freq': fs,
        't0': 0,
        'prf': None,
        'signal_duration': None,
        'delays': ul_delays,
        'sound_speed': c,
        'sequence_elements': sequence,
    }

    data_info = {
        'data_shape': (1, n_angles, n_elements, time_samples),
        'data_type': np.float32,
        'is_iq': False,
    }

    #Reshape the Mitsuba simulated data to Ultraspy's expected format
    ultraspy_data = channel_data[np.newaxis, :, :, :].astype(np.float32)

    #Manually create a 'reader-like' object to pass data to Ultraspy
    class CustomUltraspyReader:
        def __init__(self, data, data_info, acquisition_info, probe):
            self.data = data
            self.data_info = data_info
            self.acquisition_info = acquisition_info
            self.probe = probe

    reader = CustomUltraspyReader(ultraspy_data, data_info, acquisition_info, probe)

    #Initialize the DAS Beamformer
    beamformer = DelayAndSum(on_gpu=False) # Explicitly CPU
    beamformer.automatic_setup(reader.acquisition_info, reader.probe)

    d_data = reader.data[0]

    print(beamformer)

    #Define a Scan Region of Interest
    x_min_scan = -0.04
    x_max_scan = 0.04
    z_min_scan = 0.001
    z_max_scan = 0.035


    wavelength = c / fc
    step_axial = wavelength / 4
    step_lateral = wavelength / 4

    x_scan = np.arange(x_min_scan, x_max_scan + step_lateral, step_lateral)
    z_scan = np.arange(z_min_scan, z_max_scan + step_axial, step_axial)

    if len(x_scan) == 0:
        x_scan = dr.array([x_min_scan])
    if len(z_scan) == 0:
        z_scan = dr.array([z_min_scan])

    num_x_pixels = len(x_scan)
    num_z_pixels = len(z_scan)

    scan = GridScan(x_scan, z_scan)

    #Perform Beamforming
    d_output = beamformer.beamform(d_data, scan)
    d_envelope_raw = beamformer.compute_envelope(d_output, scan)

    bmode = d_envelope_raw.astype(np.float32)

    #Manual Log Compression and Display Image Generation
    bmode_db = 20 * np.log10(bmode + 1e-12)

    dynamic_range = 60
    max_db = np.max(bmode_db)
    min_db = max_db - dynamic_range
    bmode_db_clipped = np.clip(bmode_db, min_db, max_db)
    display_image = (bmode_db_clipped - min_db) / dynamic_range

    depth_axis_mm = (np.arange(time_samples) / fs) * (c / 2) * 1e3


    display_image = display_image.T

    print("Shape of display_image:", display_image.shape)
    print("Expected shape: (len(z_scan), len(x_scan)) = ({}, {})".format(len(z_scan), len(x_scan)))

    #Plotting
    plt.figure(figsize=(10, 8))

    extent = [x_scan[0] * 1e3, x_scan[-1] * 1e3, z_scan[-1] * 1e3, z_scan[0] * 1e3]

    im = plt.imshow(display_image, extent=extent, cmap='gray', origin='upper', vmin=0, vmax=1)
    plt.xlabel('Lateral (mm)') # Corrected label
    plt.ylabel('Axial/Depth (mm)')   # Corrected label
    plt.title('Simulated Ultrasound B-mode Image')
    plt.colorbar(im, label='Relative Echo Intensity (Normalized)')
    plt.gca().invert_yaxis() # This is crucial to have depth increase downwards
    plt.tight_layout()
    plt.show()

    # Debugging
    print("Raw envelope min:", np.min(d_envelope_raw), "max:", np.max(d_envelope_raw))
    print("Log values min:", np.min(bmode_db), "max:", np.max(bmode_db))
    print("After dynamic range min:", np.min(bmode_db_clipped), "max:", np.max(bmode_db_clipped))
    print("Display image min:", np.min(display_image), "max:", np.max(display_image))


    return scene, display_image


scene, display_image = us_render()


### Automatic Differentiation
initial_image = display_image

# Choose Parameters we want to optimize
params = mi.traverse(scene)
print(params.keys())

opt_key = 'shape.bsdf.roughness'

# Save original value
param_ref = params[opt_key]

# Change value and update
params[opt_key] = 0.1
params.update()

# Setting up optimizer
opt = mi.ad.Adam(lr=0.05)
opt[opt_key] = params[opt_key]

# Helper function for MSE of image to ref image
def mse(image):
    return dr.mean(dr.square(image - initial_image))

iteration_count = 5

# Gradient descent
errors = []
for it in range(iteration_count):
    # Render image
    scene, image = us_render()

    # Evaluate losee function
    loss = mse(image)

    # np to dr
    loss = dr.llvm.Float(loss)

    # Backwardspropigate through the rendering process
    dr.backward(loss)

    # Optimizer step
    opt.step()

    # Ensure opt variable stays clamped
    opt[opt_key] = dr.clip(opt[opt_key], 0.0001, 1)

    # Update scene state
    params.update(opt)

    # Track the difference between current and reference
    err_ref = dr.sum(dr.square(param_ref - params[opt_key]))

    print(f"Iteration {it:02d}: parameter error = {err_ref}", end='\r')
    errors.append(dr.slice(err_ref))
print('\nOptimization complete.')

plt.plot(errors)
plt.xlabel('Iteration'); plt.ylabel('MSE(param)'); plt.title('Parameter error plot');
plt.show()






