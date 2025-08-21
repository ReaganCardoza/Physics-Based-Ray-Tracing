import torch 
import drjit as dr
import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.nn.functional as F
import cupy as cp
from pathlib import Path
import csv
import pandas as pd

# Import Ultraspy modules
from ultraspy.beamformers.das import DelayAndSum
from ultraspy.scan import GridScan
from ultraspy.probes.factory import build_probe



import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

dr.set_flag(dr.JitFlag.VCallRecord, True)   # leave True if you *need* polymorphic calls
dr.set_flag(dr.JitFlag.LoopRecord,  True)   # leave True if you *need* dr.while_loop
dr.set_flag(dr.JitFlag.KernelHistory, False)

mi.set_variant("llvm_ad_mono")

from CustomIntegrator import UltraIntegrator
mi.register_integrator("ultrasound_integrator", UltraIntegrator)

from CustomSensor import UltraSensor
mi.register_sensor("ultrasound_sensor", UltraSensor)

from CustomEmmitter import CustomEmitter
mi.register_emitter('ultrasound_emitter', CustomEmitter)

from CustomBSDF import UltraBSDF
mi.register_bsdf('ultrasound_bsdf', UltraBSDF)







# Shows basic ability going drjit to torch to drjit

@dr.wrap(source="drjit", target="torch")
@dr.syntax
def torch_pow_2_plus_one(x):
    result = torch.pow(x,2) + 1
    return result

Float = dr.llvm.ad.Float

x = Float([1, 2, 3, 4])
dr.enable_grad(x)

y = torch_pow_2_plus_one(x)
dr.enable_grad(y)
loss = dr.sum(y)
dr.backward(loss)

print("y:", y)
print("x.grad_correct:", x.grad) 


# Lets do loops now 
@dr.wrap(source="drjit", target="torch")
@dr.syntax
def torch_while_and_for_loops(x):
    result = 0
    for i in range(2):
        j = 0
        while j < 3:
            result += torch.mul(x, i + j)
            j += 1
    return result

x_loop = Float([1, 2, 3, 4])
dr.enable_grad(x_loop)

y_loop = torch_while_and_for_loops(x_loop)
dr.enable_grad(y_loop)
loss_loop = dr.sum(y_loop)
dr.backward(loss_loop)

print("y_loop_correct:", y_loop)
print("x_loop.grad_correct:", x_loop.grad) 

# Lets try Ultraspy now

scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'ultrasound_integrator',
        'max_depth': 10, # Keep at 1 for simpler debugging
        'sampling_rate': 50e6,
        'frequency': 5e6,
        'sound_speed': 1540,
        'attenuation': 0.2, # Keep at 0.0 for debugging signal strength
        'wave_cycles': 5,
        'main_beam_angle': 8,
        'cutoff_angle': 10,
        'n_elements': 64, # Keep low for faster debugging
        'pitch': 0.00003 * 4,
        'time_samples': 10002, # Keep large enough
        'angles': dr.linspace(mi.Float, -10, 10, 5) # Keep low for faster debugging
    },
        'sensor': {
        'type': 'ultrasound_sensor',
        'num_elements_lateral': 1280,
        'elements_width': 0.003,
        'elements_height': 0.01,
        'pitch': 0.0003,
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
            'type': 'sphere',
            'center': [0, 0, 0.01],
            'radius': 0.005,
            'bsdf': {
                'type': 'ultrasound_bsdf',
                'impedance': 7.8,
                'roughness': 0.9,
                'pdf_max' : 0.2
            }
        }

}

scene = mi.load_dict(scene_dict)
integrator = scene.integrator()

n_angles = integrator.n_angles
n_elements = integrator.n_elements
time_samples = integrator.time_samples

channel_buf = dr.arange(dtype= dr.llvm.ad.Float, start=0, stop= n_angles*n_elements*time_samples)
dr.enable_grad(channel_buf)
delays_input = dr.arange(dtype= dr.llvm.ad.Float, start=0, stop= n_angles*n_elements)


@dr.wrap(source="drjit", target="torch")
@dr.syntax
def UltraSpy_Render( channel_buf, delays, visualize=False):

    channel_buf_torch = channel_buf
    delays_torch = delays


    scene = mi.load_dict(scene_dict)
    integrator = scene.integrator()
    n_angles = integrator.n_angles
    n_elements = integrator.n_elements
    time_samples = integrator.time_samples
    fs = integrator.fs
    fc = integrator.frequency
    c = integrator.sound_speed
    angles_deg = integrator.angles.numpy()


    channel_data = channel_buf_torch.reshape((n_angles, n_elements, time_samples))
    ul_delays    = delays_torch.reshape((n_angles, n_elements))

    assert isinstance(channel_data, torch.Tensor) and channel_data.requires_grad, "Input returned a non-differentiable tensor"

    #Ultraspy Integration Starts Here

    #Prepare Probe Information
    probe = build_probe(
        geometry_type='linear',
        nb_elements=n_elements,
        pitch=integrator.pitch,
        central_freq=fc,
        bandwidth=70
    )

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
    #ultraspy_data = channel_data[np.newaxis, :, :, :]
    ultraspy_data = channel_data.unsqueeze(0) 


    #r = torch.autograd.grad(torch.norm(ultraspy_data), raw_r, retain_graph=True)

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
    # d_data.requires_grad_(True)

    #print(beamformer)

    #Define a Scan Region of Interest
    x_min_scan = -0.04
    x_max_scan = 0.04
    z_min_scan = 0.001
    z_max_scan = 0.05


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
    assert isinstance(d_output, torch.Tensor) and d_output.requires_grad, "DAS returned a non-differentiable tensor"

    d_envelope_raw = beamformer.compute_envelope_torch(d_data, scan) # CHANGE BACK!!!
    assert isinstance(d_envelope_raw, torch.Tensor) and d_envelope_raw.requires_grad, "Envelope lost grad"


    bmode = d_envelope_raw


    #Manual Log Compression and Display Image Generation
    bmode_db = 20 * torch.log10(bmode + 1e-12)
    dynamic_range = 60
    max_db = torch.max(bmode_db)
    min_db = max_db - dynamic_range
    bmode_db_clipped = torch.clip(bmode_db, min_db, max_db)
    display_image = (bmode_db_clipped - min_db) / dynamic_range

    depth_axis_mm = (torch.arange(time_samples) / fs) * (c / 2) * 1e3


    display_image = display_image.transpose(-2, -1)
    #display_image.requires_grad_(True)
    assert display_image.requires_grad, "Display image lost grad"

    display_image_torch = display_image
    #display_image_torch.requires_grad_(True)

    print("acquire -> channel_buf_torch.requires_grad:", channel_buf_torch.requires_grad)

    print("after reshape -> channel_data.requires_grad:", channel_data.requires_grad)

    print("d_data.requires_grad:", d_data.requires_grad)

    print("beamform -> d_output.requires_grad:", isinstance(d_output, torch.Tensor) and d_output.requires_grad)

    print("envelope -> d_envelope_raw.requires_grad:", isinstance(d_envelope_raw, torch.Tensor) and d_envelope_raw.requires_grad)

    print("display_image.requires_grad:", display_image.requires_grad)



    params = mi.traverse(scene)
    rough_t = params['shape.bsdf.roughness']
    imp_t = params['shape.bsdf.impedance']


    return display_image_torch

def loss_energy(integ):
    """
    Simple scalar: mean squared amplitude of the simulated channel buffer.
    Pure Dr.Jit → safe to call dr.backward(loss).
    """
    buf = integ.channel_buf            # drjit Float (1D)
    return dr.mean(buf * buf)          # scalar drjit Float

@dr.wrap(source="drjit", target="torch")
@dr.syntax
def ultraspy_stage(channel_buf, delays,
                   n_angles, n_elements, time_samples, fs, fc, c, pitch,
                   visualize=False):
    # ---- Force plain Python dtypes immediately ----
    n_angles     = int(n_angles)
    n_elements   = int(n_elements)
    time_samples = int(time_samples)
    fs, fc, c    = float(fs), float(fc), float(c)
    pitch        = float(pitch)

    # Dr.Jit -> Torch (wrapper does the conversion); use only Torch/NumPy from here
    channel = channel_buf.reshape((n_angles, n_elements, time_samples))
    ul_delays = delays.reshape((n_angles, n_elements))

    probe = build_probe(geometry_type='linear',
                        nb_elements=n_elements,
                        pitch=pitch,
                        central_freq=fc,
                        bandwidth=70)

    elems = np.arange(n_elements)
    sequence = {
        'emitted':  np.tile(elems, (n_angles, 1)),
        'received': np.tile(elems, (n_angles, 1)),
    }

    acquisition_info = {
        'sampling_freq': fs,
        't0': 0.0,
        'prf': None,
        'signal_duration': None,
        'delays': ul_delays,     # torch tensor is fine
        'sound_speed': c,
        'sequence_elements': sequence,
    }

    data_info = {
        'data_shape': (1, n_angles, n_elements, time_samples),
        'data_type': np.float32,
        'is_iq': False,
    }

    d = channel.unsqueeze(0)  # [1, A, E, T]

    class Reader:
        def __init__(self, data, dinfo, ainfo, probe):
            self.data = data
            self.data_info = dinfo
            self.acquisition_info = ainfo
            self.probe = probe

    reader = Reader(d, data_info, acquisition_info, probe)
    bf = DelayAndSum(on_gpu=False)
    bf.automatic_setup(reader.acquisition_info, reader.probe)

    step = (c / fc) / 4.0
    x_scan = np.arange(-0.04,  0.04 + step, step)
    z_scan = np.arange( 0.001, 0.05 + step, step)
    scan = GridScan(x_scan, z_scan)

    # Beamform + envelope (keep it all Torch)
    d_out = bf.beamform(reader.data[0], scan)
    env = bf.compute_envelope_torch(d_out, scan)

    # Log-compress -> display image [Z, X]
    bmode_db = 20.0 * torch.log10(env + 1e-12)
    dyn = 60.0
    mx = torch.max(bmode_db)
    mn = mx - dyn
    img = (torch.clamp(bmode_db, mn, mx) - mn) / dyn
    img = img.transpose(-2, -1)

    # Return a scalar so Dr.Jit gets a simple loss
    return img.mean()




display_image = UltraSpy_Render(channel_buf, delays_input, False)
dr.enable_grad(display_image)
loss = dr.sum(display_image)
dr.backward(loss)

print("loss:", loss)
print("x.grad_correct:", channel_buf.grad) 
