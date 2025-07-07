import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

#mi.set_variant("llvm_ad_rgb")

class CustomEmitter(mi.Emitter):
    def __init__(self, props):
        super().__init__(props)

        self.number_of_elements = props.get('number_of_elements', 100)
        self.pitch = props.get('pitch', 0.001) #element center to center spacing
        self.radius = props.get('radius', 0.0) #infinite if linear
        self.element_width = props.get('element_width', .001) #the width of the element in the x direction
        self.opening_angle = dr.deg2rad(props.get('opening_angle', 0.0))
        self.steering_angle = dr.deg2rad(props.get('steering_angle', 0.0))

        #Ultrasound parameters
        self.speed_of_sound = props.get('speed_of_sound', 1540) #speed of sound through tissue
        self.central_frequency = props.get('central_frequency', 5e6) #measured in Hz of the transmiting frequency

        #Controls
        self.seed = props.get('seed', 0)

        self.element_positions = dr.zeros(mi.Point3f, self.number_of_elements)
        self.element_normals = dr.zeros(mi.Vector3f, self.number_of_elements)
        self.compute_element_geometry()

        #flags and ID
        self._flags = mi.EmitterFlags.Surface | mi.EmitterFlags.SpatiallyVarying
        self._id = props.id()

    def compute_element_geometry(self):


        #Computing the element positions first for both a linear and convex trandsucer
        opening_angle_half = self.opening_angle * 0.5

        #calculate theta increments
        theta_steps = dr.linspace(dr.scalar.ArrayXf,opening_angle_half, -opening_angle_half, self.number_of_elements)
        total_span_x = (self.number_of_elements - 1) * self.pitch

        if self.radius > 0:

            #calculate the x and y coordinates
            xs = self.radius * dr.sin(theta_steps)
            ys = self.radius * dr.cos(theta_steps)

            #calculate the normals
            nx = dr.sin(theta_steps)
            ny = dr.cos(theta_steps)

        else:
            xs = dr.linspace(dr.scalar.ArrayXf, (-total_span_x / 2), (total_span_x / 2), self.number_of_elements)
            ys = np.zeros_like(theta_steps)

            nx = np.zeros_like(theta_steps)
            ny = np.ones_like(theta_steps)

        self.element_positions = mi.Point3f(xs, ys, 0.0)
        self.element_normals = mi.Vector3f(nx, ny, 0.0)

        return self.element_positions, self.element_normals

    def sample_position(self, time, sample, active =True):

        pdf_sample_position = np.full_like(sample, (1/self.number_of_elements))

        index = dr.floor(sample * self.number_of_elements)
        index = dr.minimum(index, self.number_of_elements - 1)

        ps = mi.PositionSample3f()
        ps.p = dr.gather(mi.Point3f, self.element_positions, index)
        ps.n = dr.gather(mi.Vector3f, self.element_normals, index)
        ps.time = time

        return ps, pdf_sample_position

    def needs_sample_1(self):
        return False

    def needs_sample_2(self):
        return True

    def needs_sample_3(self):
        #firring one ray per element
        return False

    def sample_ray(self, time, sample1, sample2, sample3, active =True):
        #pick an element
        ps, pdf_pos = self.sample_position(time, sample2, active )

        psi = self.steering_angle
        direction = mi.Vector3f(dr.sin(psi), dr.cos(psi), 0)

        time_dr = np.full_like(ps.p.x, time)

        #compute the time delay per element
        time_delay = -(ps.p.x * dr.sin(psi)) / self.speed_of_sound
        t0 = time_dr + time_delay

        #Origin of the ray whcih is the element selected
        origin = mi.Point3f(ps.p)

        #spawn the ray origin, direction, and how far it goes, time
        ray = mi.Ray3f(origin, direction, t0)

        #directivity weight from paper
        fd = dr.maximum((dr.dot(direction, ps.n)), 0)

        weight = fd/ pdf_pos

        spec = mi.UnpolarizedSpectrum(weight)

        return ray, spec

    def pdf_ray(self, ray: mi.Ray3f, active: bool=True):
        return dr.full(ray.time, 1.0 / self.number_of_elements)

    def traverse(self, callback):
        callback.put_parameter('number_of_elements', self.number_of_elements, mi.ParamFlags.Differentiable)
        callback.put_parameter('pitch', self.pitch, mi.ParamFlags.Differentiable)
        callback.put_parameter('radius', self.radius, mi.ParamFlags.Differentiable)
        callback.put_parameter('opening_angle', self.opening_angle, mi.ParamFlags.Differentiable)
        callback.put_parameter('steering_angle', self.steering_angle, mi.ParamFlags.Differentiable)
        callback.put_parameter('speed_of_sound', self.speed_of_sound, mi.ParamFlags.Differentiable)
        callback.put_parameter('central_frequency', self.central_frequency, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        super().parameters_changed(keys)
        self.compute_element_geometry()








'''

if __name__ == '__main__':


    props_emitter_curved = mi.Properties()
    props_emitter_curved['number_of_elements'] = 5
    props_emitter_curved['pitch'] = .001
    props_emitter_curved['opening_angle'] = 0.5
    props_emitter_curved['radius'] = .01

    emitter = CustomEmitter(props_emitter_curved)
    positions, normals = emitter.compute_element_geometry()

    print("\nCalculated Positions:")
    pos_x, pos_y, pos_z = positions.x, positions.y, positions.z
    for i in range(len(pos_x)):
        print(f"  Element {i}: X={pos_x[i]:.4f}, Y={pos_y[i]:.4f}, Z={pos_z[i]:.4f}")

    print("\nCalculated Normals:")
    norm_x, norm_y, norm_z = normals.x, normals.y, normals.z
    for i in range(len(norm_x)):
        print(f"  Element {i}: NX={norm_x[i]:.4f}, NY={norm_y[i]:.4f}, NZ={norm_z[i]:.4f}")


##### tesing the sample positon method

    props = mi.Properties()
    props['number_of_elements'] = 5
    props['pitch'] = 0.001
    props['radius'] = 0.01
    props['opening_angle'] = 0.5
    props['steering_angle'] = 0.0
    props['speed_of_sound'] = 1540
    props['central_frequency'] = 5e6

    emitter = CustomEmitter(props)
    n = emitter.number_of_elements

    samples = dr.linspace(mi.Float, 0.0, 1.0 - 1e-6, n)

    ps, pdf = emitter.sample_position(0.0, samples)

    # 7) Extract component arrays
    xs = ps.p.x
    ys = ps.p.y
    nxs = ps.n.x
    nys = ps.n.y
    pdfs = pdf

    # 8) Print a table
    print(f"{'idx':>3} | {'x':>8} {'y':>8} | {'nx':>8} {'ny':>8} | {'pdf':>8}")
    print("-" * 50)
    for i, (x, y, nx, ny, p) in enumerate(zip(xs, ys, nxs, nys, pdfs)):
        print(f"{i:3d} | {x:8.4f} {y:8.4f} | {nx:8.4f} {ny:8.4f} | {p:8.4f}")


### Testing the sample ray method

    def ensure_list(arr, n):
        try:
            data = arr.numpy().tolist()
        except Exception:
            try:
                data = arr.tolist()
            except Exception:
                data = [float(arr)]
        if len(data) == 1 and n > 1:
            data = data * n
        return data
    base_props = mi.Properties()
    base_props['number_of_elements']  = 100
    base_props['pitch']               = 0.001  # 1 mm spacing
    base_props['radius']              = 10   # 10 mm curvature
    base_props['opening_angle']       = 3.14    # radians
    base_props['speed_of_sound']      = 1540  # m/s
    base_props['central_frequency']   = 5e6   # Hz

    # Angles to visualize (negative and positive steering)
    steering_angles = [0, 0.8]
    results = []

    # 3) For each steering angle, sample rays and collect t0 and weights
    for angle in steering_angles:
        props = mi.Properties(base_props)
        props['steering_angle'] = angle
        emitter = CustomEmitter(props)
        n = emitter.number_of_elements

        # uniform sample2 values
        sample2 = dr.linspace(mi.Float, 0.0, 1.0 - 1e-6, n)
        rays, specs = emitter.sample_ray(0.0, None, sample2, None)

        # extract arrays
        ox = ensure_list(rays.o.x, n)
        t0 = ensure_list(rays.time, n)
        w  = ensure_list(getattr(specs, 'x', specs), n)

        # convert time to microseconds
        t0_us = [val * 1e6 for val in t0]
        results.append((angle, ox, t0_us, w))

    # 4) Plot delays for both angles
    plt.figure()
    for angle, ox, t0_us, _ in results:
        plt.plot(ox, t0_us, marker='o', label=f'angle={angle:.2f} rad')
    plt.xlabel('Element X Position (m)')
    plt.ylabel('Emission Delay (µs)')
    plt.title('Plane Wave Steering Delays for Two Angles')
    plt.legend()
    plt.grid()

    # 5) Plot directivity weights for both angles
    plt.figure()
    for angle, ox, _, w in results:
        plt.plot(ox, w, marker='o', label=f'angle={angle:.2f} rad')
    plt.xlabel('Element X Position (m)')
    plt.ylabel('Weight (cosθ/pdf)')
    plt.title('Directivity Weights for Two Steering Angles')
    plt.legend()
    plt.grid()

    plt.show()

'''