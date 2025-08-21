import mitsuba as mi

import drjit as dr



#mi.set_variant("cuda_ad_mono")



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

        to_world_prop = props.get('to_world', mi.ScalarTransform4f())

        if hasattr(to_world_prop, 'matrix'):

            self.transform = mi.Transform4f(to_world_prop.matrix)

        else:

            self.transform = to_world_prop



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



        # Random offsets within element

        offset_x = (aperture_sample.x - 0.5) * self.element_width

        offset_y = (aperture_sample.y - 0.5) * self.element_height



        # local origin

        origin_local = mi.Point3f(element_x + offset_x, offset_y, element_z)



        # 3D Directional sampling 

        if hasattr(mi.warp, 'square_to_uniform_hemisphere'):

            # Sample forward hemisphere for realistic ultrasound beam pattern

            direction_local = mi.warp.square_to_uniform_hemisphere(aperture_sample)

        else:

            # Fall back to manual spherical coordinates

            phi = position_sample.y * 2 * dr.pi

            cos_theta = wavelength_sample

            theta = dr.acos(cos_theta)



            sin_theta = dr.sin(theta)

            direction_local = mi.Vector3f(sin_theta * dr.cos(phi),

                                          sin_theta * dr.sin(phi),

                                          cos_theta) # Forward hemisphere (positive Z)



        # Transform to world

        origin_world = self.transform @ origin_local

        direction_world = dr.normalize(self.transform @ direction_local)



        # Phase

        phase = 2 * dr.pi * self.center_frequency * time



        # Directivity weighting

        cos_beam_angle = direction_local.z

        directivity_weight = dr.abs(cos_beam_angle) * self.directivity





        weight = dr.cos(2 * dr.pi * self.center_frequency * time) * directivity_weight

        return mi.Ray3f(origin_world, direction_world), weight





    def traverse(self, callback):


        pass