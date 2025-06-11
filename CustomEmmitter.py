import drjit as dr
import mitsuba as mi
import math

mi.set_variant("cuda_ad_rgb")

class UltraRayEmitter(mi.Emitter):
    def __init__(self, props):
        super().__init__(props)

        # Transducer geometry parameters
        self.num_elements_lateral = props.get("num_elements_lateral", 128)
        self.elements_width = props.get("elements_width", 0.003)
        self.elements_height = props.get("elements_height", 0.01)
        self.pitch = props.get("pitch", 0.00035)
        self.radius = props.get("radius", dr.inf)

        # Emission properties
        self.speed_of_sound = props.get("speed_of_sound", 1540)
        self.center_frequency = props.get("center_frequency", 5e6)
        self.intensity = props.get("intensity", mi.Color3f(1.0))

        # Handle angles
        raw_angles = props.get("plane_wave_angles_degrees", [0.0])
        if isinstance(raw_angles, (int, float)):
            raw_angles = [float(raw_angles)]
        elif isinstance(raw_angles, tuple):
            raw_angles = list(raw_angles)

        # Convert Python list to Dr.Jit array using mi.Float constructor
        self.plane_wave_angles_rad = mi.Float(raw_angles) * dr.pi / 180.0
        self.num_plane_wave_angles = len(raw_angles)

        # Scene transformation
        self.to_world = props.get("to_world", mi.ScalarTransform4f())

        # Element positions and normals
        self.element_position_x = dr.zeros(mi.Float, self.num_elements_lateral)
        self.element_position_z = dr.zeros(mi.Float, self.num_elements_lateral)
        self.element_normals = [mi.Vector3f(0, 0, 1)] * self.num_elements_lateral

        if math.isinf(self.radius):  # Linear array
            total_width_array = (self.num_elements_lateral - 1) * self.pitch
            start_x = -total_width_array / 2
            self.element_position_x = start_x + dr.arange(mi.Float, self.num_elements_lateral) * self.pitch
            self.element_position_z = dr.zeros(mi.Float, self.num_elements_lateral)
            self.element_normals = [mi.Vector3f(0, 0, 1)] * self.num_elements_lateral
        else:  # Convex array
            theta = (dr.arange(mi.Float, self.num_elements_lateral) - self.num_elements_lateral / 2) * (self.pitch / self.radius)
            self.element_position_x = self.radius * dr.sin(theta)
            self.element_position_z = self.radius * (1 - dr.cos(theta))
            self.element_normals = [mi.Vector3f(dr.cos(t), 0.0, dr.sin(t)) for t in theta]

        # Flags and ID
        self.m_flags = mi.EmitterFlags.Surface | mi.EmitterFlags.SpatiallyVarying
        self.m_id = props.id()

    def sample_ray(self, time, sample1, sample2, sample3, active=True):
        # Sample wavelength
        wavelengths, spec_weight = mi.sample_rgb_spectrum(sample1)

        # Element and angle selection
        element_index = dr.minimum(dr.floor(sample2.x * self.num_elements_lateral), self.num_elements_lateral - 1)
        angle_index = dr.minimum(dr.floor(sample3.y * self.num_plane_wave_angles), self.num_plane_wave_angles - 1)
        steering_angle = dr.gather(mi.Float, self.plane_wave_angles_rad, angle_index)

        # Random offsets on element surface
        offset_x = (sample2.y - 0.5) * self.elements_width
        offset_y = (sample3.x - 0.5) * self.elements_height

        x_e = dr.gather(mi.Float, self.element_position_x, element_index)
        z_e = dr.gather(mi.Float, self.element_position_z, element_index)
        origin_local = mi.Point3f(x_e + offset_x, offset_y, z_e)

        # Steering direction in xz plane
        direction_local = mi.Vector3f(dr.sin(steering_angle), 0, dr.cos(steering_angle))

        # Element normal
        normal_local = self.element_normals[element_index]

        # Transform to world
        origin_world = self.to_world @ origin_local
        direction_world = dr.normalize(self.to_world @ direction_local)

        # Directivity
        cos_theta = dr.maximum(dr.dot(direction_local, normal_local), 0.0)
        directivity_weight = (1.0 / self.num_elements_lateral) * cos_theta

        # Time delay
        delay_time = -(x_e * dr.sin(steering_angle)) / self.speed_of_sound

        # Final ray
        ray = mi.RayDifferential3f(
            o=origin_world,
            d=direction_world,
            time=time + delay_time,
            wavelengths=wavelengths
        )

        # Return ray, weighted spectrum, and activity mask
        spectrum = self.intensity * directivity_weight * spec_weight
        return ray, spectrum, active

    def traverse(self, callback):
        callback.put_parameter("intensity", self.intensity)
        callback.put_parameter("to_world", self.to_world)

    def parameters_changed(self):
        pass

    def flags(self):
        return self.m_flags

    def id(self):
        return self.m_id