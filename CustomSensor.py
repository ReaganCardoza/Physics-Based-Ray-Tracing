import mitsuba as mi
import drjit as dr

mi.set_variant('llvm_ad_rgb')

class UltraSensor(mi.Sensor):
    def __init__(self, props):
        super().__init__(props)

        # Sensor Type
        self.type = 'distant'

        # Sensor origin default location
        self.o_x = 0
        self.o_y = 0
        self.o_z = 0

        # Sensor default location
        self.r_x = 0
        self.r_y = 0
        self.r_z = 0

        # Sensor default orientation
        self.n_x = 0
        self.n_y = 0
        self.n_z = 0 

        # Scene Transform
        self.to_world = mi.ScalarTransform4f().look_at(
                origin=[self.o_x, self.o_y, self.o_z],
                target=[self.r_x, self.r_y, self.r_z],
                up=[0,0,1]
            )
        
        # Target
        self.target = [self.n_x, self.n_y, self.n_z]

        # Filter
        filter_dict = {
            'type' : 'tent',
            'radius': 0.5,
        }

        # Film
        film_dict = {
            'type':'hdrfilm',
            'width': 1,
            'height': 1,
            'filter': filter_dict
        }

 

        # Internal Definition of the Sensor, Film
        self._internal = mi.load_dict({
            # Sensor 
            'type' : self.type,
            'to_world' : self.to_world,
            'target' : self.target,

            # Film 
            'film': film_dict
            


        })

        

    # Change Sensor Origin
    def translate_origin(self, new_o_x, new_o_y, new_o_z):
        self.o_x = new_o_x
        self.o_y = new_o_y
        self.o_z = new_o_z

    # Moves the sensor to a different location
    def translate_sensor(self, new_r_x, new_r_y, new_r_z):
        self.r_x = new_r_x
        self.r_y = new_r_y
        self.r_z = new_r_z

    # Rotates the sensor
    def rotate_sensor(self, new_n_x, new_n_y, new_n_z):
        self.n_x = new_n_x
        self.n_y = new_n_y
        self.n_z = new_n_z

    #### FILL THESE OUT!!!!
    def sample_ray(self, time, wavelength_samples, sample2, sample3, reparam, active=True):
        return self._internal.sample_ray(time, wavelength_samples, sample2, sample3, reparam, active)
    
    def pdf_ray(self, ray, active=True):
        return self._internal.pdf_ray(ray, active)


        
