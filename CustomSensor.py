import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

class UltraSensor(mi.Sensor):
    def __init__(self, props):
        super().__init__(props)

        ### Declaring parameters
        self.frequency = props.get('frequency', 5e6) # 5 MHz center frequency
        self.elements = props.get('elements', 128)  # Transducer elements
        self.impulse = props.get('impulse', mi.Spectrum(1.0))
        self.transform = props.get('to_world', mi.ScalarTransform4f())

        # Acoustic properties
        self.sound_speed = props.get('sound_speed', 1540) # m/s
        self.attenuation = props.get('attenuation', 0.5)  # dB/cm/MHz


    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        # Plane wave emission pattern
        directions = mi.Vector3f(0, 0, 1) # +Z forward direction
        origins = mi.Point3f(0,0,0)       # Transducer surface

        # Create coherent pulse with phase information
        phase = 2 * dr.pi * self.frequency * time
        weight = self.impulse * dr.exp(1j * phase)

        return mi.Ray3f(origins, directions), weight