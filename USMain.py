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


# Torch → Dr.Jit → Mitsuba (mono-safe)
# Dr.jit to torch 
# FUCK DRJIT 
@dr.wrap(source='drjit', target='torch')
@dr.syntax
def simulate(scene, raw_r, raw_z):
    # Map Torch -> Dr.Jit (keeps AD)
    rough_u = dr.clip(1.0 / (1.0 + dr.exp(-raw_r)), 1e-4, 1.0 - 1e-4)
    imp_u   = dr.maximum(dr.exp(raw_z),             1e-4)

    # Reduce to width-1 AD scalars
    rough_s = dr.mean(rough_u)
    imp_s   = dr.mean(imp_u)

    params = mi.traverse(scene)

    # Splat onto packet dtype (mi.Float is packet under llvm_ad_rgb)
    rough_pkt = rough_s + dr.zeros(mi.Float)
    imp_pkt   = imp_s   + dr.zeros(mi.Float)

    # ✅ assign the mapped *packet* values (NOT raw_r/raw_z)
    params['shape.bsdf.roughness'] = rough_pkt
    params['shape.bsdf.impedance'] = imp_pkt
    params.update()

    integ = scene.integrator()
    integ.simulate_acquisition(scene)

    # Debug widths
    p = mi.traverse(scene)
    print("widths:", dr.width(p['shape.bsdf.roughness']),
                    dr.width(p['shape.bsdf.impedance']))

    return integ.channel_buf, integ.transmission_delays_buf





def us_render(scene, visualize=False, raw_r=None, raw_z=None):
   # --- 2) Get torch tensors from the wrapped simulate() ---
    if (raw_r is not None) and (raw_z is not None):
        channel_buf_torch, delays_torch = simulate(scene, raw_r, raw_z)
        assert isinstance(channel_buf_torch, torch.Tensor) and channel_buf_torch.requires_grad, "channel_buf lost grad"

    else:
        # For target image creation (no gradients)
        # You can still call simulate with torch.no_grad() or use static scene params
        with torch.no_grad():
            channel_buf_torch, delays_torch = simulate(scene, torch.tensor(0.0).requires_grad_(True), torch.tensor(0.0).requires_grad_(True))






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

    d_envelope_raw = beamformer.compute_envelope_torch(d_output, scan)
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








    if visualize:
        display_image = display_image.numpy()
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

        # # Debugging
        # print("Raw envelope min:", np.min(d_envelope_raw), "max:", np.max(d_envelope_raw))
        # print("Log values min:", np.min(bmode_db), "max:", np.max(bmode_db))
        # print("After dynamic range min:", np.min(bmode_db_clipped), "max:", np.max(bmode_db_clipped))
        # print("Display image min:", np.min(display_image), "max:", np.max(display_image))

    params = mi.traverse(scene)
    rough_t = params['shape.bsdf.roughness']
    imp_t = params['shape.bsdf.impedance']


    return display_image_torch, rough_t, imp_t



def append_loss_csv(csv_path, it, loss, roughness=None, impedance=None):
    """
    Append one row to a CSV log. Creates the file with header if missing.
    loss can be a float or torch tensor.
    """
    csv_path = Path(csv_path)
    is_new = not csv_path.exists()
    loss_val = float(loss.detach().cpu() if hasattr(loss, "detach") else loss)
    row = [int(it), loss_val,
           float(roughness.detach().cpu()) if hasattr(roughness, "detach") else (None if roughness is None else float(roughness)),
           float(impedance.detach().cpu()) if hasattr(impedance, "detach") else (None if impedance is None else float(impedance))]
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["iter", "loss", "roughness", "impedance"])
        w.writerow(row)

def plot_normalized_loss(log, save_path=None, show=True):
    """
    Plot L/L0 over iterations.
    `log` can be a list/array of losses OR a path to the CSV written by append_loss_csv.
    """
    if isinstance(log, (str, Path)):
        df = pd.read_csv(log)
        y = df["loss"].to_numpy(dtype=float)
        x = df["iter"].to_numpy(dtype=int)
    else:
        y = np.asarray(log, dtype=float)
        x = np.arange(len(y))

    if y.size == 0:
        print("No losses to plot.")
        return

    L0 = y[0]
    y_norm = y / (L0 if L0 != 0 else 1.0)

    plt.figure()
    plt.plot(x, y_norm)            # no explicit colors/styles
    plt.xlabel("Iteration")
    plt.ylabel("L / L0")
    plt.title("Normalized Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close()
# --- Setup (keep mono) ---
mi.set_variant("llvm_ad_mono")
dr.set_flag(dr.JitFlag.VCallRecord, True)
dr.set_flag(dr.JitFlag.LoopRecord,  True)
dr.set_flag(dr.JitFlag.KernelHistory, True)

scene = mi.load_dict(scene_dict)
integ = scene.integrator()

# Params
params = mi.traverse(scene)
rough = params['shape.bsdf.roughness']
imp   = params['shape.bsdf.impedance']
dr.enable_grad(rough)
dr.enable_grad(imp)
params['shape.bsdf.roughness'] = rough
params['shape.bsdf.impedance'] = imp
params.update()

# Forward
integ.simulate_acquisition(scene)

# Sanity: AD types
print("channel_buf type:", type(integ.channel_buf))
print("delays_buf  type:", type(integ.transmission_delays_buf))

# ✅ Mono-safe scalar loss (your accumulator)
L = integ.channel_buf

print(integ.channel_buf)

print(L)

# Backprop (works in mono)
dr.backward(L)



# Read grads
g_r_sum = dr.sum(dr.detach(dr.grad(rough)))
g_z_sum = dr.sum(dr.detach(dr.grad(imp)))
print("∂L/∂rough (sum):", (g_r_sum))
print("∂L/∂imp   (sum):", (g_z_sum))



# --- 2) Tiny optimization loop in Dr.Jit (SGD on rough/imp) ---
# You can start from current values; here we keep them and just take a few steps.
# lr_r = mi.Float(1e-2)   # separate lrs in case scales differ
# lr_z = mi.Float(1e-2)

# n_iters = 5
# for it in range(n_iters):
#     # Clear previous grads
#     dr.set_grad(rough, mi.Float(0))
#     dr.set_grad(imp,   mi.Float(0))

#     # Write current params back into the scene
#     params['shape.bsdf.roughness'] = rough
#     params['shape.bsdf.impedance'] = imp
#     params.update()

#     # Forward
#     integ.simulate_acquisition(scene)

#     # Loss
#     L = dr.sum(integ.channel_buf * integ.channel_buf)

#     # Backward
#     dr.backward(L)

#     # Read grads safely
#     g_r = dr.grad(rough)
#     g_z = dr.grad(imp)

#     # Convert to plain Floats for printing (optional)
#     #g_r_val = float(dr.sum(dr.detach(g_r)))
#     #g_z_val = float(dr.sum(dr.detach(g_z)))

#     # SGD update (keep parameters in a sane range)
#     rough = dr.clamp(rough - lr_r * g_r, 1e-4, 1.0 - 1e-4)
#     imp   = dr.maximum(imp   - lr_z * g_z, 1e-4)

#     # Log

#     print(f"[{it:02d}] L={float(dr.detach(L).numpy()):.6e} ")


# # Load a scene you’ll optimize against
# opt_scene = mi.load_dict(scene_dict)

# # TEMP: a simulate wrapper that DOESN'T return bb (so it can't leak a grad)
# @dr.wrap(source='torch', target='drjit')
# @dr.syntax
# def simulate_no_bb(scene, raw_r, raw_z):
#     rough = dr.clip(1.0 / (1.0 + dr.exp(-raw_r)), mi.Float(1e-4), mi.Float(1.0 - 1e-4))
#     imp   = dr.maximum(dr.exp(raw_z), mi.Float(1e-4))
#     params = mi.traverse(scene)
#     params['shape.bsdf.roughness'] = rough
#     params['shape.bsdf.impedance'] = imp
#     params.update()
#     integ = scene.integrator()
#     integ.simulate_acquisition(scene)
#     return integ.channel_buf, integ.transmission_delays_buf

# # --- Explicit check: gradient ONLY through channel_buf ---
# raw_r = torch.tensor(0.3).logit().requires_grad_(True)
# raw_z = torch.tensor(5.0).log().requires_grad_(True)

# ch, delays = simulate_no_bb(opt_scene, raw_r, raw_z)
# probe = ch.float().pow(2).mean()

# gr = torch.autograd.grad(probe, raw_r, retain_graph=True, allow_unused=True)[0]
# gz = torch.autograd.grad(probe, raw_z, allow_unused=True)[0]   # last one can drop the graph
# print("channel-only grad wrt raw_r:", gr)  # expect tensor(nonzero) if path exists
# print("channel-only grad wrt raw_z:", gz)



# scene_dict['shape']['bsdf']['roughness'] = 0.9
# scene_dict['shape']['bsdf']['impedance'] = 7.8
# ref_scene = mi.load_dict(scene_dict)
# #with torch.no_grad():
# target_img, _, _ = us_render(ref_scene, visualize=True, raw_r=torch.tensor(0.5), raw_z=torch.tensor(7.8).log())

# # Optim vars (unconstrained)
# raw_r = torch.tensor(0.3, requires_grad=True) # if target is ~0.3 roughnes
# raw_z = torch.tensor(5.0, requires_grad=True)    # if target is ~5.0 impedance
# raw_r.requires_grad_(True)
# raw_z.requires_grad_(True)
# opt = torch.optim.Adam([raw_r, raw_z], lr=1e-2)

# opt_scene = mi.load_dict(scene_dict)

# for it in range(100):
#     opt.zero_grad()

#     sim_img, r_map, z_map = us_render(opt_scene, visualize=False, raw_r=raw_r, raw_z=raw_z)
#     loss = F.mse_loss(sim_img.float(), target_img.float())
#     print("Calculated Loss")

#     # Log the *mapped* params
#     #append_loss_csv("loss_log.csv", it, loss, r_map, z_map)
#     print("Saved Loss")

#     loss.backward()
#     print("grads:", raw_r.grad, raw_z.grad)

#     opt.step()

#     if it % 1 == 0:
#         print(f"[{it:03d}] loss={loss.item():.6f}")

# --- Gradient smoke test for simulate(..) ---

# def assert_has_grad(y, xs, name="y"):
#     """Fail fast if y doesn't carry grad back to each x in xs."""
#     probe = y.float().pow(2).mean()          # scalar torch loss
#     grads = torch.autograd.grad(probe, xs, retain_graph=True, allow_unused=True)
#     for i, (x, g) in enumerate(zip(xs, grads)):
#         if g is None:
#             raise RuntimeError(f"No grad from {name} to xs[{i}] (check wrapper/broadcast).")
#     return grads
# def run_simulate_grad_test(scene_dict):
#     scene = mi.load_dict(scene_dict)

#     # Use multi-lane inputs
#     N = 8
#     raw_r = torch.full((N,), 0.4,        dtype=torch.float32, requires_grad=True)
#     raw_z = torch.full((N,), np.log(7.8), dtype=torch.float32, requires_grad=True)

#     # PacketOps can be left ON now; turning it OFF is optional
#     ch, delays = simulate(scene, raw_r, raw_z)

#     integ = scene.integrator()
#     exp_size = int(integ.n_angles) * int(integ.n_elements) * int(integ.time_samples)
#     assert ch.numel() == exp_size

#     loss = torch.norm(ch)
#     loss.backward()

#     print("[simulate-test] loss:", float(loss.detach()))
#     print("[simulate-test] ∂loss/∂raw_r:", raw_r.grad)
#     print("[simulate-test] ∂loss/∂raw_z:", raw_z.grad)


# # ---- run it ----
# run_simulate_grad_test(scene_dict)
