import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')

class UltraSensor(mi.Sensor):
    def __init__(self, props):
        super().__init__(props)

        ### Declaring parameters
        self.frequency = props.get('frequency', 5e6) # 5 MHz center frequency
        self.elements = props.get('elements', 128)  # Transducer elements
        self.transform = props.get('to_world', mi.ScalarTransform4f())
        self.sound_speed = props.get('sound_speed', 1540) # m/s

        # Store emission time for phase reference
        self.emission_time = mi.Float(0)

        # For reception sensitivity
        self.directivity = props.get('directivity', 1.0)

    '''
    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        # Store emission time for phase reference
        self.emission_time = time

        # element positions (for linear array)
        x = (position_sample.x - 0.5) * (self.elements * 0.3e-3) # 0.3mm pitch
        origin = mi.Point3f(x, 0, 0)

        # Plane wave emission direction
        direction = mi.Vector3f(0, 0, 1)

        # initial phase (transmission)
        phase = 2 * dr.pi * self.frequency * time
        weight = mi.Spectrum(dr.cos(phase), dr.sin(phase), 0.0)

        return mi.Ray3f(origin, direction), weight
    '''
    
    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        # Simple perspective rays pointing toward sphere
        origin = mi.Point3f(0, 0, 3)  # Fixed position behind sphere
        
        # Map position_sample to image plane
        x = (position_sample.x - 0.5) * 2  # [-1, 1]
        y = (position_sample.y - 0.5) * 2  # [-1, 1]
        direction = dr.normalize(mi.Vector3f(x, y, -1))  # Point toward -Z
        
        weight = mi.Spectrum(1.0, 0.0, 0.0)
        return mi.Ray3f(origin, direction), weight

    
    def sample_direction(self, it, sample, active=True):
        # Reception phase calculation (echo detection)
        time_delta = it.time - self.emission_time
        distance = time_delta * self.sound_speed
        phase = 2 * dr.pi * self.frequency * time_delta

        # Directivity sensitivity
        cos_theta = dr.dot(it.wi, mi.Vector3f(0,0,1))
        sensitivity = self.directivity * dr.abs(cos_theta)

        # Complex reception weight
        weight = mi.Spectrum(dr.cos(phase), dr.sin(phase), 0.0) * sensitivity

        return mi.DirectionSample3f(
            d=it.wi,
            pdf=1.0,
            time=it.time,
            wavelength=mi.Float(0),
            weight=weight
        ), True
