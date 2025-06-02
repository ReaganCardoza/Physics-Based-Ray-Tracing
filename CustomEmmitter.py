import drjit as dr
import mitsuba as mi

class UltraRayEmitter(mi.Emitter):
    def __init__(self, props):
        super().__init__(props)

        #Transducer Geometry default values
        self.num_elements_lateral = props.get("num_elements_lateral", 128)
        self.elements_width = props.get("elements_width", 0.003) #0.3mm
        self.elements_height = props.get("elements_height", 0.01) #0.1mm
        self.pitch = props.get("pitch", 0.00035) #0.35mm
        self.radius =props.get("radius", dr.inf) #Default is linear which has infinite radius

        #Emmission properties
        self.speed_of_sound = props.get("speed_of_sound", 1540) #m/s

        #Lower frequencies go deeper into the material but high frequency give better resolution
        self.center_frequency = props.get("center_frequency", 5e6) #5MHz

        # Base emitted pressure/intensity per ray, scaled by directivity
        self.intensity = props.get("intensity", mi.Color3f(1.0))

        #Plane wave imaging parameters
        raw_input_angles = props.get('plane_wave_angles_degrees', [0.0])

        plane_wave_angles = []
        if not isinstance(raw_input_angles, list):
            for angles in raw_input_angles:
                plane_wave_angles.append(float(angles))

        #Convert the angles into drjit compatible floats
        self.plane_wave_angles_rad = dr.zeros(mi.Float, len(plane_wave_angles))
        for i, angle_deg in enumerate(plane_wave_angles):
            self.plane_wave_angles_rad[i] = float(angle_deg) * dr.pi / 180.0
        self.num_plane_wave_angles = len(self.plane_wave_angles_rad)

        #Emmiter transformations set in Scene
        self.to_world = props.get("to_world", mi.ScalarTransform4f())


        #Element Positions in Local Transducer space
        #assuming 1D linear array centered at x = 0 and transducer surface is at z=0 emitting towards positive z

        self.element_position_x = dr.zeros(mi.Float, self.num_elements_lateral)
        if self.num_elements_lateral > 1:
            total_width_array = (self.num_elements_lateral -1 ) * self.pitch
            start_x = -total_width_array / 2
            # dr.arrange can create a sequence, useful if pitch is also an array for some reason
            self.element_position_x = start_x + dr.arange(mi.float, self.num_elements_lateral) * self.pitch

        else: #Single element at origin
            self.element_position_x[0] = 0.0

        #The normal vector for each element is in the positive z direction
        self.element_normal_local = mi.Vector3f(0, 0, 1)

        #Mitsuba Emitter Flags describes the type of emitter to Mitsuba for potential optimizations
        # mi.EmitterFlags.Surface emits from a surface
        # mi.EmitterFlags.SpatiallyVarying emission characteristics can change over the surface
        self.m_flags = mi.EmitterFlags.Surface | mi.EmitterFlags.SpatiallyVarying

        #debugging tool to store ID
        self.m_id = props.id()


        #Spawning the rays
    def sample_ray(self, time, wavelength_sample, sample1, sample2, active=True):

        #pick a randome transducer element
        element_index = dr.minimum(dr.floor(sample1 * self.num_elements_lateral), self.num_elements_lateral - 1)

        #Select a plane wave steering angle (positive tilt rigth and negative tilt left)
        angle_index = dr.minimum(dr.floor(sample2.x * self.num_plane_wave_angles), self.num_plane_wave_angles - 1)
        steering_angle = dr.gather(mi.float, self.plane_wave_angles_rad, angle_index)

        #compute the origin (x, y, z)
        origin_local = mi.Point3f(dr.gather(mi.Float, self.element_position_x, element_index), 0, 0)

        #Compute the direction
        direction_local = mi.Vector3f(dr.sin(steering_angle), 0, dr.cos(steering_angle)) #steering in the xz plane

        #Transfrom both to world and space
        origin_world = self.to_world @ origin_local
        #Normalize direction
        direction_world = dr.normalize(self.to_world @ direction_local)

        #Emmission directivity fd = (omega dot n)
        cos_theta = dr.dot(direction_local, self.element_normal_local)
        directivity_weight = (dr.maximum(cos_theta, 0.0))#there is no negative weight

        #normalizing 1/N
        normalized_directivity = 1 / self.num_elements_lateral

        #weighted directivity
        weighted_directivity = normalized_directivity * directivity_weight


        #Emmission Time Delay for ray timing te based on steering angle
        #xe is the lateral position of the elment, steering angle theta, speec of sound c
        x_e = dr.gather(mi.Float, self.element_position_x, element_index)
        delay_time = -(x_e * dr.sin(steering_angle)) / self.speed_of_sound

        #Create ray
        ray = mi.Ray(o=origin_world, d=direction_world, time=time + delay_time, wavelength=wavelength_sample)

        return ray, weighted_directivity
















