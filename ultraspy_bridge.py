# ultraspy_bridge.py  (NEW FILE)
import drjit as dr
import torch
import numpy as np
from ultraspy.beamformers.das import DelayAndSum
from ultraspy.scan import GridScan
from ultraspy.probes.factory import build_probe

@dr.wrap(source="drjit", target="torch")
@dr.syntax
def ultraspy_stage(channel_buf, delays,
                   n_angles, n_elements, time_samples, fs, fc, c, pitch,
                   visualize=False):
    # force plain Python scalars
    n_angles     = int(n_angles);  n_elements   = int(n_elements)
    time_samples = int(time_samples)
    fs, fc, c    = float(fs), float(fc), float(c)
    pitch        = float(pitch)

    # Dr.Jit->Torch happens automatically
    channel   = channel_buf.reshape((n_angles, n_elements, time_samples))
    ul_delays = delays.reshape((n_angles, n_elements))

    probe = build_probe('linear', n_elements, pitch, fc, bandwidth=70)

    elems = np.arange(n_elements)
    sequence = {'emitted':  np.tile(elems, (n_angles, 1)),
                'received': np.tile(elems, (n_angles, 1))}

    acquisition_info = {
        'sampling_freq': fs, 't0': 0.0, 'prf': None, 'signal_duration': None,
        'delays': ul_delays, 'sound_speed': c, 'sequence_elements': sequence
    }
    data_info = {'data_shape': (1, n_angles, n_elements, time_samples),
                 'data_type': np.float32, 'is_iq': False}

    d = channel.unsqueeze(0)

    class Reader:
        def __init__(self, data, dinfo, ainfo, probe):
            self.data, self.data_info = data, dinfo
            self.acquisition_info, self.probe = ainfo, probe

    reader = Reader(d, data_info, acquisition_info, probe)
    bf = DelayAndSum(on_gpu=False)
    bf.automatic_setup(reader.acquisition_info, reader.probe)

    step = (c / fc) / 4.0
    x_scan = np.arange(-0.04,  0.04 + step, step)
    z_scan = np.arange( 0.001, 0.05 + step, step)
    scan = GridScan(x_scan, z_scan)

    d_out = bf.beamform(reader.data[0], scan)
    env = bf.compute_envelope_torch(d_out, scan) if hasattr(bf, 'compute_envelope_torch') else torch.abs(d_out)

    bmode_db = 20.0 * torch.log10(env + 1e-12)
    dyn = 60.0; mx = torch.max(bmode_db); mn = mx - dyn
    img = (torch.clamp(bmode_db, mn, mx) - mn) / dyn
    img = img.transpose(-2, -1)
    return img  # or img.mean() if you want a scalar
