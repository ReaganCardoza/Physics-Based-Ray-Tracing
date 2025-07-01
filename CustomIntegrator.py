import mitsuba as mi
import drjit as dr

#mi.set_variant("cuda_ad_mono")

class UltraIntegrator(mi.SamplingIntegrator):
    def __init__(self, props):
        super().__init__(props)
        # Scene independent ray tracing parameters
        self.max_depth = props.get('max_depth', 2)      # One round trip
        self.frequency = props.get('frequency', 5e6)    # Hz
        self.sound_speed = props.get('sound_speed', 1540) # m/s
        self.attenuation = props.get('attenuation', 0.5) # dB/cm/MHz
        self.wave_cycles = props.get('wave_cycles', 5)
        self.main_beam_angle = props.get("main_beam_angle", 5) # deg
        self.cutoff_angle = props.get("cutoff_angle", 120) # deg
        self.fs = props.get('sampling_rate', 50e6) # Hz

        # Transducer geometry
        self.n_elements = props.get('n_elements', 128)
        self.pitch = props.get('pitch', 0.00035)
        self.elem_x = self.pitch * (dr.arange(mi.Float, self.n_elements) - (self.n_elements - 1)/2)
        self.elem_pos = mi.Vector3f(self.elem_x, 0, 0)
        self.trans_norm = mi.Vector3f(0, 0, 1)

        # Plane Wave transmission
        self.angles = dr.linspace(mi.Float, -30, 30, 25) 
        self.n_angles = len(self.angles)


        # Per ray initial state constants
        self.init_amp = 1.0
        self.init_atten = 1.0
        self.init_tof = 0.0

        # Echo accumulation buffer
        self.time_samples = props.get('time_samples', 3000)
        self.channel_buf = dr.zeros(mi.Float, self.n_angles * self.n_elements * self.time_samples)

        # House keeping list for post-p
        self.rx_counter = dr.zeros(mi.UInt32, self.n_angles * self.n_elements)
        



    
    def sample(self, scene, sampler, ray, medium, active=True):

        ray = mi.Ray3f(ray)

        # Alias
        n_elem = self.n_elements
        n_angle = self.n_angles
        fs = self.fs
        c = self.sound_speed

        # Pre ray initial state
        angle_id = mi.UInt32(dr.floor(sampler.next_1d() * self.n_angles))
        elem_id  = mi.UInt32(dr.floor(sampler.next_1d() * self.n_elements))
        angle_rad = (-30. + angle_id * (60. / (n_angle - 1))) * dr.pi / 180.0
        x_elem = self.pitch * (mi.Float(elem_id) - (n_elem - 1) * 0.5)
        tx_delay = (x_elem * dr.sin(angle_rad)) / c

        # Running state
        amp = mi.Color1f(self.init_amp)
        atten = mi.Float(self.init_atten)
        tof = mi.Float(self.init_tof)
        geo_len = mi.Float(0.0)
        depth = mi.UInt32(0)

        ### Debugging
        '''
        print("Element positions (x):", self.elem_x.numpy())
        print("Steering angles (deg):", self.angles.numpy())
        origin = mi.Point3f(x_elem, 0, 0)
        direction = mi.Vector3f(dr.sin(angle_rad), 0, dr.cos(angle_rad))
        print("Ray origins:", origin.numpy())
        print("Ray directions:", direction.numpy())
        print("TX delays (us):", tx_delay.numpy() * 1e6)
        '''

        # Helper function
        def directivity_weight_i(wi, n, alpha_m, alpha_c):
            # incoming angle
            alpha = dr.abs(dr.acos(dr.dot(n, wi)))
            dr.print(alpha_m)
            dr.print(alpha)
            dr.print(alpha_c)
            mid_cond = (alpha_c - alpha) / (alpha_c - alpha_m)
            return dr.select(alpha <= alpha_m, 1.0,
                dr.select(alpha <= alpha_c, 
                            mid_cond,
                            0.0))


        # Symbolic while loop
        state = (amp, atten, tof, geo_len, depth, ray, active)

        def cond(amp, atten, tof, geo_len, depth, ray, active):
            return active
        
        def body(amp, atten, tof, geo_len, depth, ray, active):
            ### Primary intersection
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            distanace = dr.select(active, si.t, 0.0)

            ### Targeted secondary ray to random elemt
            elem_off = (n_elem - 1) * 0.5
            target = mi.Point3f(self.pitch * (mi.Float(elem_id) - elem_off), 0, 0)
            sec_dir = dr.normalize(target - si.p)
            vis_si = scene.ray_intersect(si.spawn_ray(sec_dir), active)
            visible = vis_si.is_valid()

            ### Attenuation, TOF and Phase
            atten *= dr.exp(-self.attenuation * self.frequency * 1e-6 * distanace / 8.686)
            tof += distanace / self.sound_speed
            phase = 2 * dr.pi * self.frequency * tof

            ### BSDF Interaction
            ctx = mi.BSDFContext()
            bsdf = si.bsdf()
            bs, a_resp = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_1d(), active)
            amp *= a_resp

            ### Scatter echo into channel buffer
            mask = active & visible
            fd = directivity_weight_i(sec_dir, si.sh_frame.n, dr.deg2rad(self.main_beam_angle), dr.deg2rad(self.cutoff_angle))
            dr.print(fd)

            # Safegaurds
            total_time = dr.maximum(tx_delay + tof, 0.0) # non-negative time

            # Clamped time index
            t_idx = dr.round(total_time * fs)
            t_idx = dr.clamp(t_idx, 0, self.time_samples - 1)
            t_idx = mi.UInt32(t_idx)

            # Clamped flat index
            channel_index = angle_id * n_elem + elem_id
            flat = channel_index * self.time_samples + t_idx
            max_flat = self.n_angles * self.n_elements * self.time_samples - 1
            flat = dr.clamp(flat, 0, max_flat)

            pressure_scalar = atten * amp * fd * dr.sin(phase)
            dr.scatter_reduce(dr.ReduceOp.Add, self.channel_buf, pressure_scalar, flat, active=mask)

            ### Spawn new rays
            new_dir = si.to_world(bs.wo)
            ray = si.spawn_ray(dr.normalize(new_dir))

            ### Stopping criteria
            geo_len += distanace
            depth += 1

            cos_min = dr.cos(dr.deg2rad(self.cutoff_angle))
            within_angle = dr.dot(-ray.d, self.trans_norm) >= cos_min
            path_ok = geo_len < 0.2
            depth_ok = depth < self.max_depth

            # Russian roulette
            rr_prob = dr.minimum(dr.max(atten * amp), 0.95)
            survive = (sampler.next_1d() < rr_prob) & active

            active &= within_angle & path_ok & depth_ok & survive
            atten = dr.select(survive, atten / rr_prob, 0.0)


            return (amp, atten, tof, geo_len, depth, ray, active)


        amp, atten, tof, geo_len, depth, ray, active = dr.while_loop(state, cond, body)

        return mi.Color1f(0.0), active, []    

        