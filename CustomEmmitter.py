import mitsuba as mi
import drjit as dr
import numpy as np

from RayTracingV0 import element_idx

mi.set_variant('cuda_ad_rgb')

'''
File found in mitsuba3/src/emmitters
The custom emmiter needs to to include: 

pressure rays instead of photons.

Add physically realistic timing (delays per element).

Incorporate transducer geometry and directionality.

'''
class customEmitter(mi.Emmiter):

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










