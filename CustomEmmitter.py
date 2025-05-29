import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant('cuda_ad_rgb')

'''
File found in mitsuba3/src/emmitters
The custom emmiter needs to to include: 

pressure rays instead of photons.

Add physically realistic timing (delays per element).

Incorporate transducer geometry and directionality.

'''
class CustomEmitter(mi.Emitter):

    def __init__(self, props):
        mi.Emmiter.__init__(self, props)

        #Defining the emmiter perameters base cases
        self.num_elements = props.get('num_elements', 128)
        self.element_width = props.get('element_width', 0.0003) #Meters
        self.element_height = props.get('element_height', 0.01) #Meters
        self.pitch = props.get('pitch', 0.00035) #meters - center to center spacing
        self.radius = props.get('radius', 0.0) #meters - set to 0 for linear array

        #Emmision parameters base case from degrees to radians
        plane_wave_angles_degrees = props.get('plane_wave_angles_degrees', [-15, 0, 15])
        if isinstance(plane_wave_angles_degrees, (float, int)):
            plane_wave_angles_degrees = [float(plane_wave_angles_degrees)]
        else:
            plane_wave_angles_degrees = [float(angle) for angle in plane_wave_angles_degrees]

        self.plane_wave_angles_radians = [np.radians(angle) for angle in plane_wave_angles_degrees]

        #Convert to Dr.Jit array
        if self.plane_wave_angles_radians:
            self.plane_wave_angles_radians_jit = mi.Float(self.plane_wave_angles_radians)
        else:
        #if the plane wave is empty in angles then just defualt to 0 degrees
            self.plane_wave_angles_radians = [0.0]
            self.plane_wave_angles_radians_jit = mi.Float([0.0])

        #Number of angles
        self.num_angles = len(self.plane_wave_angles_radians)

        #Acoustic Parameters
        #Intensity and speed of sound
        #Intensity - could be good to adjust to change how things render at deeper depths
        #it is the intiial pressure as it is being emmited before the weighting
        #Speed of sound - Depending the medium we are rendering through this can be adjusted
        self.intensity = props.get('intensity', 1.0) #base pressyre/amplitude
        self.speed_of_sound = props.get('speed_of_sound', 1540.0) #m/s (this is for tissue)

        #Calculating the area of the entire transducer that is emitting
        self.total_area = float(self.height * self.width * self.num_elements)
        #check to make sure valid height, width, and number of elemetns have been entered correclty
        if self.total_area <= 0.0:
            raise ValueError('Total area must be positive')

        #precompute the center posititon and normal vector of each transducer element in its space
        #x-axis is the lateral direction
        #y-axis is the elevation direction
        #z-axis is the axial direction that is emmiting forward

        #initialia the Dr.Jit arrays
        self.element_centers_local = dr.zeros(mi.Point3f, self.num_elements)
        self.elements_normals_local = dr.zeros(mi.Vector3f, self.num_elements)

        centers_list = []
        normals_list = []

        #Linear Case
        if self.radius == 0.0:
            for i in range(self.num_elements):
                x_position = (i - (self.num_elements -1) / 2) * self.pitch #calculate the middle index for even spacing
                centers_list.append(mi.Point(x_position, 0, 0))
                normals_list.append(mi.Vector3f(0, 0, 1))
        #Curved transducer (curving in XZ plane)
        else:
            if self.radius <= 0.0:
                raise ValueError('Radius must be positive')
            for i in range(self.num_elements):
        #The center is the negative of the z direction of the distance radius
        #the arc lentgh is calcualting the same
        #so the angle is arc lentgh / radius
                angle = ((i - (self.num_elements -1) / 2) * self.pitch) / self.radius
        #Need to double check this math
                center_x = self.radius * np.sin(angle)
                center_z = self.radius * (np.cos(angle) - 1)

                self.element_centers_local[i] = mi.Point3f(center_x, 0, center_z)
        #Normalize the local normal vector for unti lentgh and cosine weighting
                normal_vec = dr.normalize(mi.Vector3f(center_x, 0, center_z + self.radius))
                normals_list.append(normal_vec)


        #populating the Jit arrays
        if self.num_elements > 0:
            current_variant_float = mi.float()

            #element centers
            centers_flat =[]
            for p in centers_list: centers_flat.extend([p.x, p.y, p.z])
            if centers_flat:
                self.element_centers_local = mi.Vector3f(current_variant_float(centers_flat))

            #Elements normals
            normals_flat = []
            for n in normals_list: normals_flat.extend([n.x, n.y, n.z])
            if normals_flat:
                self.normals_local = mi.Vector3f(current_variant_float(normals_flat))



        #setting up the flagse class mitsuba.EmitterFlags from api
        #Surface because it is emmited from a surface which we define as the transducer
        #SpatiallyVarying delays applied to form the plane wave ins
        self.m_flags = mi.EmitterFlags.Surface | mi.EmmiterFlags.SpatiallyVarying

    def primary_ray(self, time, sample_origin_u, sample_origin_v, sample_angle_idx_float, active):

        #Select transducer element and point on element
        elem_idx_float = sample_origin_u * self.num_elements

        elem_idx_int_type = dr.int32_array_t(elem_idx_float)  # Get the JIT int type
        elem_idx = elem_idx_int_type(dr.floor(elem_idx_float))  # Cast result of floor

        min_val_idx = elem_idx_int_type(0)
        max_val_idx = elem_idx_int_type(self.num_elements - 1)
        elem_idx = dr.clip(elem_idx, min_val_idx, max_val_idx)

        u_on_element = elem_idx_float - dr.floor(elem_idx_float)
        v_on_element = sample_origin_v

        # Gather using the JIT integer index array
        elem_center_l = dr.gather(mi.Point3f, self.element_centers_local, elem_idx, active)
        elem_normal_l = dr.gather(mi.Vector3f, self.element_normals_local, elem_idx, active)

        bitangent_l = mi.Vector3f(0, 1, 0)
        tangent_l = dr.normalize(dr.cross(bitangent_l, elem_normal_l))

        origin_offset_l = tangent_l * (u_on_element - 0.5) * self.element_width + \
                          bitangent_l * (v_on_element - 0.5) * self.element_height

        ray_origin_l = elem_center_l + origin_offset_l
        point_normal_l = elem_normal_l

        ray_origin_w = self.m_to_world.transform_affine(ray_origin_l)
        point_normal_w = dr.normalize(self.m_to_world.transform_normal(point_normal_l))

        # Select plane wave steering angle
        angle_idx_float_scaled = sample_angle_idx_float * self.num_angles

        angle_idx_int_type = dr.int32_array_t(angle_idx_float_scaled)
        angle_idx = angle_idx_int_type(dr.floor(angle_idx_float_scaled))

        min_val_angle_idx = angle_idx_int_type(0)
        # Ensure num_angles-1 is not negative if num_angles is 0 (though __init__ prevents this)
        max_val_angle_idx = angle_idx_int_type(dr.maximum(0, self.num_angles - 1))
        angle_idx = dr.clip(angle_idx, min_val_angle_idx, max_val_angle_idx)

        steer_angle = mi.Float(0.0)
        if self.num_angles > 0:
            steer_angle = dr.gather(mi.Float, self.plane_wave_angles_rad_jit, angle_idx, active)

        dir_x_l = dr.sin(steer_angle)
        dir_z_l = dr.cos(steer_angle)
        ray_dir_l = dr.normalize(mi.Vector3f(dir_x_l, mi.Float(0.0), dir_z_l))
        ray_dir_w = dr.normalize(self.m_to_world.transform_normal(ray_dir_l))

        #Calculate emission delay
        x_coord_for_delay = ray_origin_l.x

        delay = (x_coord_for_delay * dr.sin(steer_angle)) / mi.Float(self.speed_of_sound)
        ray_time = time + delay

        # Calculate ray weight
        cosine_weight = dr.maximum(mi.Float(0.0), dr.dot(ray_dir_w, point_normal_w))

        pdf_direction_val = mi.Float(1.0)  # Default to 1.0 to avoid division by zero if num_angles is 0
        if self.num_angles > 0:
            pdf_direction_val = 1.0 / mi.Float(self.num_angles)

        # Handle potential division by zero if pdf_direction_val is still zero
        # However, __init__ ensures num_angles is at least 1.

        spectrum_val = mi.Float(self.base_intensity) * cosine_weight

        # Safe division for spectrum_weighted
        spectrum_weighted = dr.select(
            dr.abs(pdf_direction_val) > 1e-9,  # Check if pdf_direction is not effectively zero
            mi.Color3f(spectrum_val / pdf_direction_val),
            mi.Color3f(0.0)  # Return zero if pdf_direction is zero
        )

        # Construct the ray
        # mi.Color0f() should be fine for the JIT variant, but getting a warning not sure why
        ray = mi.RayDifferential3f(o=ray_origin_w, d=ray_dir_w, time=ray_time, wavelengths=mi.Color0f())

        # PDF of sampling the ray origin on the surface
        total_area_jit = mi.Float(self.total_area)
        pdf_position = dr.select(
            active & (total_area_jit > 1e-9),
            1.0 / total_area_jit,
            mi.Float(0.0)
        )

        return ray, spectrum_weighted, pdf_position

    def to_string(self):
        # Ensure plane_wave_angles_rad_py is used for string representation
        angles_deg_str = [np.degrees(angle) for angle in self.plane_wave_angles_rad_py]
        return (f"CustomEmitter[\n"
                f"  num_elements = {self.num_elements},\n"
                f"  element_width = {self.element_width:.4g},\n"
                f"  element_height = {self.element_height:.4g},\n"
                f"  pitch = {self.pitch:.4g},\n"
                f"  radius = {self.radius:.4g},\n"
                f"  plane_wave_angles_deg = {angles_deg_str},\n"
                f"  num_angles = {self.num_angles},\n"
                f"  base_intensity = {self.base_intensity:.4g},\n"
                f"  speed_of_sound = {self.speed_of_sound:.4g},\n"
                f"  total_area = {self.total_area:.4g},\n"
                f"  transform = {self.m_to_world}\n"
                f"]")

#Registration format from API docs
mi.register_emitter("ultra_ray_emitter", lambda props: CustomEmitter(props))








