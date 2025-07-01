import mitsuba as mi
import drjit as dr


class UltraIntegrator(mi.SamplingIntegrator):
    def __init__(self, props):
        super().__init__(props)
        # Scene independent ray tracing parameters
        self.max_depth = props.get('max_depth', 2)
        self.frequency = props.get('frequency', 5e6)
        self.sound_speed = props.get('sound_speed', 1540)
        self.attenuation = props.get('attenuation', 0.5)
        self.wave_cycles = props.get('wave_cycles', 5)
        self.main_beam_angle = props.get("main_beam_angle", 5)
        self.cutoff_angle = props.get("cutoff_angle", 120)
        self.fs = props.get('sampling_rate', 50e6)

        # Transducer geometry
        self.n_elements = props.get('n_elements', 128)
        self.pitch = props.get('pitch', 0.00035)
        self.elem_x = self.pitch * (dr.arange(mi.Float, self.n_elements) - (self.n_elements - 1) / 2)
        self.elem_pos = mi.Vector3f(self.elem_x, 0, 0)
        self.trans_norm = mi.Vector3f(0, 0, 1)

        # Plane Wave transmission
        self.angles = props.get('angles', dr.linspace(mi.Float, -30, 30, 25))
        self.n_angles = len(self.angles)

        # Per ray initial state constants
        self.init_amp = 1.0
        self.init_atten = 1.0
        self.init_tof = 0.0

        # Echo accumulation buffer
        self.time_samples = props.get('time_samples', 3000)
        self.channel_buf = dr.zeros(mi.Float, self.n_angles * self.n_elements * self.time_samples)

        # New: Buffer to store initial transmission delays
        self.transmission_delays_buf = dr.zeros(mi.Float, self.n_angles * self.n_elements)

        # House keeping list for post-p
        self.rx_counter = dr.zeros(mi.UInt32, self.n_angles * self.n_elements)

    def sample(self, scene, sampler, ray, medium, active=True):
        return mi.Color1f(0.0), active, []

    def simulate_acquisition(self, scene):
        n_angles_scalar = int(self.n_angles)
        n_elements_scalar = int(self.n_elements)
        fs_scalar = float(self.fs)
        c_scalar = float(self.sound_speed)
        pitch_scalar = float(self.pitch)
        time_samples_scalar = int(self.time_samples)

        self.channel_buf = dr.zeros(mi.Float, n_angles_scalar * n_elements_scalar * time_samples_scalar)
        self.transmission_delays_buf = dr.zeros(mi.Float, n_angles_scalar * n_elements_scalar)

        for angle_idx_val in range(n_angles_scalar):
            angle_id_outer = mi.UInt32(angle_idx_val)
            angle_rad_outer = self.angles[angle_idx_val] * dr.pi / 180.0

            for elem_idx_val in range(n_elements_scalar):
                elem_id_outer = mi.UInt32(elem_idx_val)
                x_elem_outer = pitch_scalar * (mi.Float(elem_id_outer) - (n_elements_scalar - 1) * 0.5)

                tx_delay_outer = mi.Float( (x_elem_outer * dr.sin(angle_rad_outer)) / c_scalar )
                delay_flat_idx_outer = mi.UInt32(angle_id_outer * n_elements_scalar + elem_id_outer)

                dr.scatter(self.transmission_delays_buf, tx_delay_outer, delay_flat_idx_outer)

                # --- REVERTED: Original ray tracing setup ---
                origin = mi.Point3f(x_elem_outer, 0, 0) # Use outer variable
                direction = mi.Vector3f(dr.sin(angle_rad_outer), 0, dr.cos(angle_rad_outer)) # Use outer variable

                sensor_transform = scene.sensors()[0].transform

                origin_world = sensor_transform @ origin
                direction_world = dr.normalize(sensor_transform @ direction)
                ray = mi.Ray3f(origin_world, direction_world)

                amp = mi.Color1f(self.init_amp)
                atten = mi.Float(self.init_atten)
                tof = mi.Float(self.init_tof)
                geo_len = mi.Float(0.0)
                depth = mi.UInt32(0)
                active_ray = mi.Bool(True)

                def directivity_weight_i(wi, n, alpha_m, alpha_c):
                    trans_normal_world = dr.normalize(sensor_transform @ mi.Vector3f(0, 0, 1))
                    alpha = dr.abs(dr.acos(dr.dot(trans_normal_world, wi)))
                    mid_cond = (alpha_c - alpha) / (alpha_c - alpha_m)
                    return dr.select(alpha <= alpha_m, 1.0,
                                     dr.select(alpha <= alpha_c,
                                               mid_cond,
                                               0.0))

                state = (amp, atten, tof, geo_len, depth, ray, active_ray)

                def cond(amp, atten, tof, geo_len, depth, ray, active_ray):
                    # Original condition for max_depth and path length
                    return active_ray & (depth < self.max_depth) & (geo_len < 0.2)

                def body(amp, atten, tof, geo_len, depth, ray, active_ray):
                    ### Primary intersection
                    si = scene.ray_intersect(ray, active_ray)
                    active_ray &= si.is_valid()
                    distance = dr.select(active_ray, si.t, 0.0)

                    elem_off = (n_elements_scalar - 1) * 0.5
                    target = mi.Point3f(pitch_scalar * (mi.Float(elem_id_outer) - elem_off), 0, 0)
                    target_world = sensor_transform @ target

                    sec_dir = dr.normalize(target_world - si.p)
                    vis_si = scene.ray_intersect(si.spawn_ray(sec_dir), active_ray)
                    visible = vis_si.is_valid() & active_ray

                    atten_factor = dr.exp(-self.attenuation * self.frequency * 1e-6 * distance / 8.686)
                    atten *= atten_factor
                    tof += distance / c_scalar
                    phase = 2 * dr.pi * self.frequency * tof

                    ctx = mi.BSDFContext()
                    bsdf = si.bsdf()
                    bs, a_resp = bsdf.sample(ctx, si, dr.full(mi.Float, 0.5), dr.full(mi.Float, 0.5), active_ray)
                    amp *= a_resp

                    # REVERTED: Use original mask logic
                    mask = active_ray & visible # Re-enable the 'visible' check

                    fd = directivity_weight_i(sec_dir, si.sh_frame.n, dr.deg2rad(self.main_beam_angle),
                                              dr.deg2rad(self.cutoff_angle))
                    # REVERTED: Use original pressure_scalar
                    pressure_scalar = atten * amp * fd * dr.sin(phase)

                    total_time = dr.maximum(tx_delay_outer + tof, 0.0)
                    t_idx = dr.round(total_time * fs_scalar)
                    t_idx = dr.clamp(t_idx, 0, time_samples_scalar - 1)
                    t_idx = mi.UInt32(t_idx)

                    channel_index_local = angle_id_outer * n_elements_scalar + elem_id_outer
                    flat = channel_index_local * time_samples_scalar + t_idx
                    max_flat = n_angles_scalar * n_elements_scalar * time_samples_scalar - 1
                    flat = dr.clamp(flat, 0, max_flat)

                    # REVERTED: Use original pressure_scalar and mask
                    dr.scatter_reduce(dr.ReduceOp.Add, self.channel_buf, pressure_scalar, flat, active=mask)

                    new_dir = si.to_world(bs.wo)
                    ray = si.spawn_ray(dr.normalize(new_dir))

                    geo_len += distance
                    depth += 1

                    trans_norm_world = dr.normalize(sensor_transform @ mi.Vector3f(0, 0, 1))
                    cos_min = dr.cos(dr.deg2rad(self.cutoff_angle))
                    within_angle = dr.dot(-ray.d, trans_norm_world) >= cos_min
                    path_ok = geo_len < 0.2
                    depth_ok = depth < self.max_depth

                    rr_prob = dr.minimum(dr.max(atten * amp), 0.95)
                    survive = (mi.Float(0.5) < rr_prob) & active_ray

                    active_ray &= within_angle & path_ok & depth_ok & survive
                    atten = dr.select(survive, atten / rr_prob, 0.0)

                    return (amp, atten, tof, geo_len, depth, ray, active_ray)

                amp, atten, tof, geo_len, depth, ray, active_ray = dr.while_loop(state, cond, body)

        print(f"Simulation complete. Channel buffer populated. Transmission delays stored.")
        return True