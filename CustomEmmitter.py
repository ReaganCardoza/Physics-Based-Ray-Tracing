import drjit as dr
import mitsuba as mi

mi.set_variant("llvm_ad_mono")
class CustomEmitter(mi.Emitter):
    def __init__(self, props):
        super().__init__(props)

        #Defining the Transducer Geometry (Properties are in Degrees and Meters)
        self.number_of_elements = props.get("number_of_elements", 64)
        self.pitch = props.get("pitch", 0.0003)
        self.element_width = props.get("element_width", 0.0003)
        self.element_height = props.get("element_height", 0.0005)
        self.radius = props.get("radius", 0.0)
        self.opening_angle = props.get("opening_angle", 0.0) #angle in degrees
        self.number_of_rays_per_element = props.get("number_of_rays_per_element", 1)
        self.number_of_total_rays = self.number_of_elements * self.number_of_rays_per_element

        #Speed of sound (m/s) and Beam Steering (deg)
        self.speed_of_sound = props.get("speed_of_sound", 1540)
        self.steering_angle_min = props.get("steering_angle_min", -10.0)
        self.steering_angle_max = props.get("steering_angle_max", 10.0)

        #Precomputing the element center positions and normals
        self.element_positions, self.element_normals = self.compute_element_geoemtry()

        self._flags = mi.EmitterFlags.Surface | mi.EmitterFlags.SpatiallyVarying
        self._id = props.id()

    def compute_element_geometry(self):

        #Computing the positons of elements and normals for the linear array
        if self.radius == 0.0:
            x = dr.linspace(dr.llvm.Float,-(self.number_of_elements - 1)/2 * self.pitch,
                            (self.number_of_elements - 1)/2 * self.pitch,
                            self.number_of_elements)
            positions = mi.Point3f(x, 0.0, 0.0)
            normals = mi.Vector3f(0.0, 0.0, 1.0)

        #Computing postions and normals for convex array
        else:
            angle_span = dr.deg2rad(self.opening_angle)
            thetas = dr.linspace(dr.llvm.Float, -angle_span/2, angle_span/2, self.number_of_elements)
            x = self.radius * dr.sin(thetas)
            z = self.radius * dr.cos(thetas)
            positions = mi.Point3f(x, 0.0, z)
            normals = mi.Vector3f(dr.sin(thetas), 0.0, dr.cos(thetas))

        return positions, dr.normalize(normals)

    def sample_position(self, time, sample, active=True):

        sample1, sample2 = sample

        #Sample a random element
        element_ind_f = dr.floor(sample1 * self.number_of_elements)
        element_idx = dr.minimum(element_ind_f, self.number_of_elements - 1).astype(dr.llvm.Float)

        #Get the position and normal of the element that was sampled
        center = dr.gather(mi.Point3f, self.element_positions, element_idx)
        normal = dr.gather(mi.Vector3f, self.element_normals, element_idx)

        #Random lateral offset in the element width
        dx = (sample2.x - 0.5) * self.element_width
        dy = (sample2.y - 0.5) * self.element_height

        #Sample position all put together
        position = center + mi.Vector3f(dx, dy, 0.0)

        ps = mi.PositionSample3f()
        ps.p = position
        ps.n = normal
        ps.time = time
        ps.delta = False

        #Uniform PDF over the surface of the transducer
        pdf = 1 / (self.number_of_elements * self.element_width * self.element_height)

        return ps, pdf

    def sample_ray(self, time, sample1, sample2, sample3, active=True):
        ps, pdf = self.sample_position(time, (sample1, sample2), active)

        #Sample a random steering angle for Plane Wave Imaging (PWI)
        psi_min = dr.deg2rad(self.steering_angle_min)
        psi_max = dr.deg2rad(self.steering_angle_max)
        psi = psi_min + sample3 * (psi_max - psi_min)

        #Direction of the vector
        direction = mi.Vector3f(dr.sin(psi), 0.0, dr.cos(psi))

        #Time delay for each element at the x positon
        time_delay = -(ps.p.x * dr.sin(psi)) / self.speed_of_sound
        delta_t = time + time_delay

        #Directivity weighting (cosine weighting) from paper
        fd = dr.maximum(0.0, dr.dot(direction, ps.n))
        weight = fd / self.number_of_total_rays

        #Spawning rays
        origin = mi.Point3f(ps.p)

        ray = mi.Ray3f(o=origin, d=direction, time=delta_t)

        spec = mi.UnpolarizedSpectrum(weight)

        return ray, spec


    def sample_ray_differential(self, *args, **kwargs):
        ray, spec = self.sample_ray(*args, **kwargs)
        return ray, spec, mi.RayDifferential3f()

    def traverse(self, callback):
        callback.put_parameter("number_of_elements", self.number_of_elements, mi.ParamFlags.Differentiable)
        callback.put_parameter("pitch", self.pitch, mi.ParamFlags.Differentiable)
        callback.put_parameter("element_width", self.element_width, mi.ParamFlags.Differentiable)
        callback.put_parameter("element_height", self.element_height, mi.ParamFlags.Differentiable)
        callback.put_parameter("radius", self.radius, mi.ParamFlags.Differentiable)
        callback.put_parameter("opening_angle", self.opening_angle, mi.ParamFlags.Differentiable)
        callback.put_parameter("steering_angle_min", self.steering_angle_min, mi.ParamFlags.Differentiable)
        callback.put_parameter("steering_angle_max", self.steering_angle_max, mi.ParamFlags.Differentiable)
        callback.put_parameter("speed_of_sound", self.speed_of_sound, mi.ParamFlags.Differentiable)
        callback.put_parameter("rays_per_element", self.number_of_rays_per_element, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        super().parameters_changed(keys)
        self.element_positions, self.element_normals = self.compute_element_geometry()
        self.number_of_rays_per_element = self.number_of_elements * self.number_of_rays_per_element