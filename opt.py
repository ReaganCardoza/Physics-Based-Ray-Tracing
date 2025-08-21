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
import gc

# Import Ultraspy modules
from ultraspy.beamformers.das import DelayAndSum
from ultraspy.scan import GridScan
from ultraspy.probes.factory import build_probe



import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

dr.set_flag(dr.JitFlag.VCallRecord, False)   # leave True if you *need* polymorphic calls
dr.set_flag(dr.JitFlag.LoopRecord,  False)   # leave True if you *need* dr.while_loop
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
        'time_samples': 1, # Keep large enough
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
                'roughness': 0.3,
                'pdf_max' : 0.1
            }
        }

}



scene = mi.load_dict(scene_dict)

@dr.wrap(source="drjit", target="torch")
@dr.syntax
def ultraspy_stage(channel_buf, delays,
                   n_angles, n_elements, time_samples, fs, fc, c, pitch,
                   visualize=False, no_grad = False):
    # ---- Force plain Python dtypes immediately ----
    n_angles     = int(n_angles)
    n_elements   = int(n_elements)
    time_samples = int(time_samples)
    fs, fc, c    = float(fs), float(fc), float(c)
    pitch        = float(pitch)

    # Dr.Jit -> Torch (wrapper does the conversion); use only Torch/NumPy from here
    channel = channel_buf.reshape((n_angles, n_elements, time_samples))
    ul_delays = delays.reshape((n_angles, n_elements))

    # # Ultraspy setup (Torch + NumPy only)
    # from ultraspy.beamformers.das import DelayAndSum
    # from ultraspy.scan import GridScan
    # from ultraspy.probes.factory import build_probe
    # import numpy as np
    # import torch

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


    return img.detach().clone() if no_grad else img


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
    y = np.asarray(log)
    x = np.arange(len(y))

    L0 = y[0]
    y_norm = y / (L0 if L0 != 0 else 1.0)

    plt.figure()
    plt.plot(x, y_norm)            # no explicit colors/styles
    plt.xlabel("Iteration")
    plt.ylabel("L / L0")
    plt.title("Normalized Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("loss.pdf", bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()

def plot_params(p_input, ref, label):
    num_itrs = len(p_input)
    itrs = np.arange(num_itrs)
    ref_list = ref * np.ones(num_itrs)

    plt.plot(itrs, p_input, label = f"{label} Current")
    plt.plot(itrs, ref_list, label = f"{label} Reference")
    plt.title(f"{label} per iteration")
    plt.xlabel("Iterations")
    plt.ylabel(f"{label}")
    plt.legend()
    plt.savefig(f"{label}_per_iteration.pdf", bbox_inches="tight", dpi=150)





def loss_energy(integ):
    buf = integ.channel_buf
    return dr.mean(buf * buf)



def show_ultraspy_image(img, name="Ultrasound.pdf", x_scan=None, z_scan=None):
    """
    Display a B-mode image (Torch or NumPy tensor) from ultraspy_stage.
    Optionally provide x/z scan axes (in meters) for proper labels.
    """
    if hasattr(img, "detach"):   # Torch → NumPy
        img = img.detach().cpu().numpy()

    plt.figure(figsize=(8,6))
    if x_scan is not None and z_scan is not None:
        extent = [x_scan[0]*1e3, x_scan[-1]*1e3,
                  z_scan[-1]*1e3, z_scan[0]*1e3]  # mm
        plt.imshow(img, cmap="gray", extent=extent,
                   origin="upper", vmin=0, vmax=1)
        plt.xlabel("Lateral (mm)")
        plt.ylabel("Depth (mm)")
    else:
        plt.imshow(img, cmap="gray", origin="upper", vmin=0, vmax=1)
        plt.xlabel("X pixels")
        plt.ylabel("Z pixels")

    plt.title("Ultraspy B-mode Image")
    plt.colorbar(label="Normalized Echo Intensity")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(name)

def make_reference_image(scene_dict):
    scene  = mi.load_dict(scene_dict)
    integ  = scene.integrator()
    params = mi.traverse(scene)

    # Write target params
    with dr.suspend_grad():  # <- Dr.Jit: do NOT record AD graph
        params['shape.bsdf.roughness'] = mi.Float(target_rough)
        params['shape.bsdf.impedance'] = mi.Float(target_imp)
        params.update()

        # Forward w/o AD tracking
        integ.simulate_acquisition(scene)
        ch = integ.channel_buf
        dl = integ.transmission_delays_buf

        img_ref = ultraspy_stage(
            ch, dl,
            int(integ.n_angles), int(integ.n_elements), int(integ.time_samples),
            float(integ.fs), float(integ.frequency), float(integ.sound_speed),
            float(integ.pitch), no_grad = True
        )

    img_ref = dr.llvm.TensorXf(img_ref)
    return img_ref

def write_params_from_opt():
    """Map raw → physical and write into the scene."""
    rough = dr.clip(opt['raw_r'], 1e-4, 1.0 - 1e-4)
    imp   = dr.maximum(dr.exp(opt['raw_z']), 1e-4)
    params['shape.bsdf.roughness'] = rough
    params['shape.bsdf.impedance'] = imp
    params.update()
    return rough, imp

def forward_loss(ref_img, iter):
    """Run Mitsuba forward then Ultraspy scalar loss (Dr.Jit scalar)."""
    rough, imp = write_params_from_opt()
    integ.simulate_acquisition(scene)
    ch = integ.channel_buf
    dl = integ.transmission_delays_buf

    # Your Torch→Dr.Jit wrapper that returns a *scalar* Dr.Jit value
    L = ultraspy_stage(
        ch, dl,
        int(integ.n_angles), int(integ.n_elements), int(integ.time_samples),
        float(integ.fs), float(integ.frequency), float(integ.sound_speed),
        float(integ.pitch)
    )

    show_ultraspy_image(L, name = f"ultrasound_{iter}.pdf")

    ultrasound_export = L.numpy()
    np.savez(f"ultrasound_{iter}.npz", ultrasound = ultrasound_export) 


    MSE = dr.mean(dr.power(L - ref_img, 2))


    return MSE, imp, rough


scene = mi.load_dict(scene_dict)
integ = scene.integrator()
params = mi.traverse(scene)





# Choose your desired target material params
target_rough = 0.50
target_imp   = 7.80



ref_img = make_reference_image(scene_dict)
#show_ultraspy_image(ref_img, name="Reference.pdf")


dr.flush_malloc_cache()


# Initialize raw vars from current scene parameters
init_rough = params['shape.bsdf.roughness']
init_imp   = params['shape.bsdf.impedance']
raw_r = mi.Float(dr.log(init_rough) - dr.log(1.0 - init_rough))  # logit
raw_z = mi.Float(dr.log(init_imp))                                # log

dr.enable_grad(raw_r)
dr.enable_grad(raw_z)

# Mitsuba/Dr.Jit Adam
opt = dr.opt.Adam(lr=1e-4)
opt['raw_r'] = raw_r
opt['raw_z'] = raw_z





loss_history = []
rough_history = []

n_steps = 10
for it in range(n_steps):
    # Reset grads on the opt variables (fresh adjoints each iter)
    dr.set_grad(opt['raw_r'], mi.Float(0))
    dr.set_grad(opt['raw_z'], mi.Float(0))

    L, imp, rough = forward_loss(ref_img, iter=it)

    loss_history.append(L.numpy())
    rough_history.append(rough.numpy())

    loss_export = np.array(loss_history)
    np.savez("loss.npz",loss=loss_export)

    rough_export = np.array(rough_history)
    np.savez("rough.npz", roughness=rough_export)


    # Backprop through Ultraspy loss 
    dr.backward(L)

    print(rough.grad)

    # Optimizer update
    opt.step()

    print(f"[{it:02d}] L={L}  "
          f"| rough≈{rough}  imp≈{imp}")

    plot_normalized_loss(loss_history)
    plot_params(rough_history, target_rough,"Roughness")







