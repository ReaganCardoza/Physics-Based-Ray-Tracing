import mitsuba as mi
import drjit as dr
import numpy as np
import math
import threading
import os
import tqdm
from concurrent.futures import ThreadPoolExecutor
import queue, threading, os
from ultraspy_bridge import ultraspy_stage
import torch

#mi.set_variant("llvm_ad_rgb")


class UltraIntegrator(mi.SamplingIntegrator):
    def __init__(self, props):
        super().__init__(props)
        # Scene independent ray tracing parameters
        self.max_depth = props.get('max_depth', 2) #amount of times the ray is allowed to bounce
        self.frequency = props.get('frequency', 5e6) #Transducer frequency
        self.sound_speed = props.get('sound_speed', 1540) #Speed of sound in the medium
        self.attenuation = props.get('attenuation', 0.5) #the gradual loss of ultrasound wave intensity as it travels through a medium, like tissue
        self.wave_cycles = props.get('wave_cycles', 5) #Need to look into this not currently being used
        self.main_beam_angle = props.get("main_beam_angle", 10) #Transducer property that is the focus increase the area of resolution
        self.cutoff_angle = props.get("cutoff_angle", 20)  #Controls how everything outside of that you don't want to pay attention to
        self.fs = props.get('sampling_rate', 50e6) #Transducer property the number of times per second that a signal is measured and converted into a digital representation

        # Transducer geometry
        self.n_elements = props.get('n_elements', 128) #Transducer property for number of element
        self.pitch = props.get('pitch', 0.00035) #Spacing in between elements
        self.elem_x = self.pitch * (dr.arange(mi.Float, self.n_elements) - (self.n_elements - 1) / 2) #Evenly positoning the elements
        self.elem_pos = mi.Vector3f(self.elem_x, 0, 0) #Placing in mi vector format
        self.trans_norm = mi.Vector3f(0, 0, 1) #Assining normals to elements for linear

        # Plane Wave transmission
        self.angles = props.get('angles', dr.linspace(mi.Float, -30, 30, 25)) #Scanining angle horizontally?
        self.n_angles = len(self.angles)

        # Per ray initial state constants
        self.init_amp = 1.0 #Modified throughout the loop
        self.init_atten = 1.0 #Modified throughout the loop
        self.init_tof = 0.0 #Starting time

        # Echo accumulation buffer
        self.time_samples = props.get('time_samples', 3000)
        self.channel_buf = dr.zeros(mi.Float, self.n_angles * self.n_elements * self.time_samples) #Initializing the array

        # New: Buffer to store initial transmission delays
        self.transmission_delays_buf = dr.zeros(mi.Float, self.n_angles * self.n_elements)

        self.ray_count = 0
        self.loss_probe = mi.Float(0.0)




    #Not sure
    def sample(self, scene, sampler, ray, medium, active=True):
        return mi.Color1f(0.0), active, []

    #Needed for the def
    def traverse(self, callback):
        pass



    def simulate_acquisition(self, scene):

        #Aliases
        n_angles_scalar = int(self.n_angles)
        n_elements_scalar = int(self.n_elements)
        fs_scalar = mi.Float(self.fs)
        c_scalar = mi.Float(self.sound_speed)
        pitch_scalar = float(self.pitch)
        time_samples_scalar = int(self.time_samples)
        num_rays = n_angles_scalar * n_elements_scalar

        #Initializing the matrix
        total_bins = n_angles_scalar * n_elements_scalar * time_samples_scalar

        # channel_buf now built via differentiable reductions (no scatter)
        self.channel_buf = dr.zeros(mi.Float, total_bins)
        self.transmission_delays_buf = dr.zeros(mi.Float, n_angles_scalar * n_elements_scalar)

        # Initializing random number generator class
        rng = dr.rng(seed=0)

        @dr.wrap(source="drjit", target="torch")
        @dr.syntax
        def bin_sum(values_f32, indices_u32, total_bins_i64):
            # 1) Torch-ify & flatten
            idx  = indices_u32.long().reshape(-1)
            vals = values_f32.reshape(-1)

            # 2) Robust broadcast: if vals is scalar but idx has many lanes, expand it
            if vals.numel() == 1 and idx.numel() > 1:
                vals = vals.expand(idx.numel())

            # 3) Allocate output and do differentiable accumulation
            out = torch.zeros(int(total_bins_i64.item()),
                            dtype=vals.dtype,
                            device=vals.device)
            out.index_add_(0, idx, vals)  # grads flow w.r.t. vals
            return out

        for angle_idx_val in range(n_angles_scalar):
            angle_id_outer = mi.UInt32(angle_idx_val)
            angle_rad_outer = self.angles[angle_idx_val] * dr.pi / 180.0

            for elem_idx_val in range(n_elements_scalar):
                elem_id_outer = mi.UInt32(elem_idx_val)

                #uniformly placing the elements
                x_elem_outer = pitch_scalar * (mi.Float(elem_id_outer) - (n_elements_scalar - 1) * 0.5)

                #Computing the time
                tx_delay_outer = mi.Float( (x_elem_outer * dr.sin(angle_rad_outer)) / c_scalar )

                #Building the index to express in one flat vector
                delay_flat_idx_outer = mi.UInt32(angle_id_outer * n_elements_scalar + elem_id_outer)


                #placing in array
                dr.scatter(self.transmission_delays_buf, tx_delay_outer, delay_flat_idx_outer)

                # --- REVERTED: Original ray tracing setup ---
                origin = mi.Point3f(x_elem_outer, 0, 0) # Use outer variable
                direction = mi.Vector3f(dr.sin(angle_rad_outer), 0, dr.cos(angle_rad_outer)) # Use outer variable

                #Moving the sensor to the same position
                sensor_transform = scene.sensors()[0].transform

                origin_world = sensor_transform @ origin
                direction_world = dr.normalize(sensor_transform @ direction)

                #Declaring the ray
                ray = mi.Ray3f(origin_world, direction_world)

                #Need to check if this is suppsoed to be color or float
                amp = mi.Float(self.init_amp)
                atten = mi.Float(self.init_atten)
                tof = mi.Float(self.init_tof)
                geo_len = mi.Float(0.0)
                depth = mi.UInt32(0)
                active_ray = mi.Bool(True)

                #rng = dr.rng()

                def directivity_weight_o(wo, n, N):
                    return mi.Float((dr.dot(wo,n)) / mi.Float(N))

                def directivity_weight_i(wi, alpha_m, alpha_c):

                    #Weighting rfom the paper
                    trans_normal_world = dr.normalize(sensor_transform @ mi.Vector3f(0, 0, 1))
                    wi = -wi
                    dot = dr.dot(trans_normal_world, wi)
                    alpha = dr.abs(dr.acos(dot))

                    mid_cond = (alpha_c - alpha) / (alpha_c - alpha_m)

                    weight = dr.select(alpha <= alpha_m, 1.0,
                                     dr.select(alpha <= alpha_c,
                                               mid_cond,
                                               0.0))

                    return mi.Float(weight)

                #state = (amp, atten, tof, geo_len, depth, ray, active_ray)

                # def cond(amp, atten, tof, geo_len, depth, ray, active_ray):
                #     # Original condition for max_depth and path length
                #     return active_ray & (depth < self.max_depth) & (geo_len < 0.2)

                #Handling the math per element per angle
                #def body(amp, atten, tof, geo_len, depth, ray, active_ray):
                    ### Primary intersection
                si = scene.ray_intersect(ray, active_ray)
                active_ray &= si.is_valid()

                #If the ray is actice give the distance if it is not then return 0 distance
                distance = dr.select(active_ray, si.t, mi.Float(0.0))

                # Random sampling
                random_float = rng.random(dr.llvm.ad.Float, 128)
                recv_elem_id = mi.UInt32(dr.llvm.ad.Int(dr.floor(random_float * n_elements_scalar)))
                elem_off = (n_elements_scalar - 1) * 0.5
                target = mi.Point3f(pitch_scalar * (mi.Float(recv_elem_id) - elem_off), 0, 0)
                target_world = sensor_transform @ target
                sec_dir = dr.normalize(target_world - si.p)
                vis_si = scene.ray_intersect(si.spawn_ray(sec_dir), active_ray)
                visible = ~vis_si.is_valid() & active_ray

                atten_factor = dr.exp(-self.attenuation * self.frequency * 1e-6 * distance / mi.Float(8.686))
                atten *= atten_factor
                
                tof_to_intersection = tof + distance / c_scalar
                dist_to_recv = dr.norm(target_world - si.p)
                total_time = tx_delay_outer + tof_to_intersection + dist_to_recv / c_scalar
                phase = 2 * dr.pi * self.frequency * total_time

                ctx = mi.BSDFContext()
                bsdf = si.bsdf()

                sample1_value = mi.Float(0.5)
                sample2_value = mi.Float(0.5)
                bs, a_resp = bsdf.sample(ctx, si, sample1_value, sample2_value, active_ray)
                cos_theta = dr.dot(si.sh_frame.n, -ray.d)  # Incoming direction relative to normal
                amp *= a_resp * cos_theta * dr.maximum(bs.pdf, 1e-6)

                #dr.print(amp)

                # REVERTED: Use original mask logic
                mask = visible & active_ray # Re-enable the 'visible' check

                fd = directivity_weight_i(sec_dir, dr.deg2rad(self.main_beam_angle), dr.deg2rad(self.cutoff_angle)) * directivity_weight_o( ray.d , si.sh_frame.n ,num_rays)
                    
                # REVERTED: Use original pressure_scalar
                pressure_scalar = atten * amp * fd * dr.sin(phase)

                #Calculating total time of flight

                t_idx = dr.round(total_time * fs_scalar)
                t_idx = dr.clamp(t_idx, 0, time_samples_scalar - 1)
                t_idx = mi.UInt32(t_idx)


                #Accumulating the channel data
                channel_index_local = angle_id_outer * n_elements_scalar + recv_elem_id
                flat = channel_index_local * time_samples_scalar + t_idx
                max_flat = n_angles_scalar * n_elements_scalar * time_samples_scalar - 1
                flat = dr.clamp(flat, 0, max_flat)

                # REVERTED: Use original pressure_scalar and mask
                # dr.scatter_reduce(dr.ReduceOp.Add, self.channel_buf, pressure_scalar, flat, active=mask,)

                # lanes = number of SIMD lanes in this step (128 in your case)
                lanes = dr.width(flat)   # == dr.width(mask)

                # Broadcast the scalar to a lane-vector
                val_vec = pressure_scalar * dr.ones(mi.Float, lanes)

                # Mask it to zero-out inactive lanes (same width as indices)
                vals = dr.select(mask, val_vec, dr.zeros(mi.Float, lanes))
                inds = flat  # mi.UInt32, same width as vals

                # Differentiable binning: returns a length=total_bins vector
                chan = bin_sum(vals, inds, mi.Int64(total_bins))

                # Accumulate (plain add is differentiable)
                self.channel_buf = self.channel_buf + chan



                new_dir = si.to_world(bs.wo)
                ray = si.spawn_ray(dr.normalize(new_dir))
                self.ray_count += 1

                geo_len += distance
                depth += 1

                trans_norm_world = dr.normalize(sensor_transform @ mi.Vector3f(0, 0, 1))
                cos_min = dr.cos(dr.deg2rad(self.cutoff_angle))
                within_angle = dr.dot(ray.d, trans_norm_world) >= cos_min
                path_ok = geo_len < 0.2
                depth_ok = depth < self.max_depth

                #possible bug that is killing rays
                rr_cut_off =  rng.uniform(dr.llvm.ad.Float,128)
                rr_prob = dr.minimum(dr.max(atten * amp), 1.0)
                survive = (mi.Float(rr_cut_off) < rr_prob) & active_ray

                active_ray &= within_angle & path_ok & depth_ok & survive
                atten = dr.select(survive, atten / rr_prob, 0.0)

                # return (amp, atten, tof, geo_len, depth, ray, active_ray)

                # #amp, atten, tof, geo_len, depth, ray, active_ray = dr.while_loop(state, cond, body)
                # amp, atten, tof, geo_len, depth, ray, active_ray = body(amp, amp, atten, geo_len, depth, ray, active_ray)
                        # Accumulate a scalar 'loss_probe' that is differentiable in mono
                # e.g., sum of squared per-ray amplitudes BEFORE scatter
                #if not hasattr(self, 'loss_probe'):
                    
                # 'pressure_scalar' is the scalar you computed just before the scatter
                # Accumulate only when you compute a contribution (mask):
                self.loss_probe += dr.sum(dr.select(mask, pressure_scalar*pressure_scalar, mi.Float(0.0)))

        print(f"Simulation complete. Channel buffer populated. Transmission delays stored.")
        print(f"{self.ray_count} spawned in simulation")




        return True #Returning True?

    def traverse(self, callback):
        callback.put('pitch', self.pitch, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass
