import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

class UltraIntegrator(mi.SamplingIntegrator):
    def __init__(self, props):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 2)      # One round trip
        self.frequency = props.get('frequency', 5e6)    # Hz
        self.sound_speed = props.get('sound_speed', 1540) # m/s
        self.attenuation = props.get('attenuation', 0.5) # dB/cm/MHz


    
    def sample(self, scene, sampler, ray, medium, active=True):
        # Initialize as Dr.Jit arrays, not scalars
        result_real = dr.zeros(mi.Float, dr.width(active))
        result_imag = dr.zeros(mi.Float, dr.width(active))
        attenuation = dr.ones(mi.Float, dr.width(active))
        tof_total = dr.zeros(mi.Float, dr.width(active))
        depth = 0

        current_impedance = 1.5e6

        while dr.any(active) and (depth < self.max_depth):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            if not dr.any(active):
                break 

            new_impedance = dr.select(active, 
                                    dr.full(mi.Float, 1.6e6, dr.width(active)),
                                    current_impedance)

            R = (new_impedance - current_impedance) / (new_impedance + current_impedance)

            sample = sampler.next_1d()
            reflect = (sample < dr.abs(R)) & active
            active &= reflect

            if not dr.any(active):
                break

            distance = si.t
            att_db = self.attenuation * self.frequency * 1e-6 * distance
            attenuation *= dr.exp(-att_db / 8.686)

            tof_total += 2 * distance / self.sound_speed
            phase = 2 * dr.pi * self.frequency * tof_total

            # Now all operations are array-to-array
            pressure_mag = attenuation * R
            result_real += pressure_mag * dr.cos(phase)
            result_imag += pressure_mag * dr.sin(phase)
            print(result_imag, result_real)

            # Reflection
            reflect_dir = ray.d - 2 * dr.dot(ray.d, si.n) * si.n
            direction = dr.select(si.is_valid() & active, reflect_dir, ray.d)
            ray = si.spawn_ray(direction)
            current_impedance = new_impedance
            depth += 1

        magnitude = dr.sqrt(result_real**2 + result_imag**2)
        return mi.Spectrum(result_real, result_imag, magnitude), active, []
