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


    '''
    def sample(self, scene, sampler, ray, medium, active=True):
        result_real = dr.zeros(mi.Float)
        result_imag = dr.zeros(mi.Float)
        attenuation = 1.0
        tof_total = 0.0 # time of flight
        depth = 0

        current_impedance = 1.5e6  # Default soft tissue
        if medium is not None:
            current_impedance = medium.get('impedance', 1.5e6)


        while dr.any(active) and (depth < self.max_depth):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()

            if not dr.any(active):
                break 

            bsdf = si.bsdf()
            

            
            new_impedance = 1.6e6

            # Calculate reflection coefficient (pressure)
            R = (new_impedance - current_impedance) / (new_impedance + current_impedance)

            # Random Termination
            sample = sampler.next_1d()
            reflect = (sample < dr.abs(R)) & active
            active &= reflect

            # Attenuation Calculation
            distance = si.t
            att_db = self.attenuation * self.frequency * 1e-6 * distance * 100 # dB
            attenuation *= dr.exp(-att_db / 8.686) # Convert dB to Neper

            # time of flight and phase
            tof_total += 2 * distance / self.sound_speed # Round trip
            phase = 2 * dr.pi * self.frequency * tof_total

            # Complex pressure contribution
            pressure_mag = attenuation * R
            result_real += pressure_mag * dr.cos(phase)
            result_imag += pressure_mag * dr.sin(phase)

            # prepare next ray (reflection)
            direction = ray.d - 2 * dr.dot(ray.d, si.n) * si.n
            ray = si.spawn_ray(direction)
            current_impedance = new_impedance # Update medium impedance

            depth += 1


        return mi.Spectrum(result_real, result_imag, 0.0), active, []
        '''
    def sample(self, scene, sampler, ray, medium, active=True):
        si = scene.ray_intersect(ray, active)
        if dr.any(si.is_valid()):
            return mi.Spectrum(1.0, 0.0, 0.0), active, []  # Return white where hit
        else:
            return mi.Spectrum(0.0, 0.0, 0.0), active, []  # Return black where miss

