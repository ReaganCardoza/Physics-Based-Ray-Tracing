import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')

class UltraSensor(mi.Sensor):
    def __init__(self, props):
        super().__init__(props)

        # Geometry parameters
        self.num_elements_lateral = props.get('num_elements_lateral', 128)  # Transducer elements
        self.element_width = props.get('elements_width', 0.003)
        self.element_height = props.get('elements_height', 0.01)
        self.pitch = props.get('pitch', 0.00035)
        self.radius = props.get('radius', dr.inf)

        # Emission properties
        self.center_frequency = props.get('center_frequency', 5e6) # 5 MHz center frequency
        self.sound_speed = props.get('sound_speed', 1540) # m/s

        # Transform
        self.transform = props.get('to_world', mi.ScalarTransform4f())


        # Store emission time for phase reference
        self.emission_time = mi.Float(0)

        # For reception sensitivity
        self.directivity = props.get('directivity', 1.0)

    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        # Store emission time for phase reference
        self.emission_time = time

        # element positions (for linear array)
        element_index = dr.minimum(dr.floor(position_sample.x * self.num_elements_lateral), self.num_elements_lateral -1)

        # Calculate element posiitons based on array geometry
        if dr.isinf(self.radius):   # Linear Array
            total_width_array = (self.num_elements_lateral - 1) * self.pitch
            start_x = -total_width_array / 2
            element_x = start_x + element_index * self.pitch
            element_z = 0.0
        else: #Convex array
            theta_element = (element_index - self.num_elements_lateral / 2) * (self.pitch / self.radius)
            element_x = self.radius * dr.sin(theta_element)
            element_z = self.radius * (1 - dr.cos(theta_element))
        

        # Plane wave emission direction
        direction = mi.Vector3f(0, 0, -1)

        # initial phase (transmission)
        phase = 2 * dr.pi * self.frequency * time
        #print(phase)
        weight = mi.Spectrum(dr.cos(phase), dr.sin(phase), 0.0)
        print(f"Emission phase: {phase}, Origin: {origin}, Weights: {weight}")

        return mi.Ray3f(origin, direction), weight        self.transform = props.get('to_world', mi.ScalarTransform4f())

    