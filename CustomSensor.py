import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

class UltraSensor(mi.Sensor):
    def __init__(self, props):
        super().__init__(props)

        ### Declaring parameters

        # Geometry and Positioning and time
        self.orientation = props.get('orientation', 0)
        self.position = props.get('position', 0)
        self.origin = props.get('origin', 0)
        self.time = props.get('time', 0)

        # Scene transformation
        self.to_world = props.get('to_world',0)

        # Sampling 
        self.samp_dir = props.get('samp_dir', 0)            # Sampling direction (omega_i) 
        self.num_sample = props.get('num_time', 0)          # Number of rays     (N) MOVING TO INTEGRATOR
        self.m_needs_sample_2 = props.get('m_needs_sample_2', False) # Only needs one ray sample
        self.m_needs_sample_3 = props.get('m_needs_sample_3', False) # Only needs one ray sample

        # Pressure Signal Components
        self.directivity = props.get('directivity', 0)      # directivity function
        self.samp_prob = props.get('samp_prob', 0)          # sampling probabilty (p_t(omega_i))
        self.p_element_i = props.get('p_element_i', 0)      # P value pre weighting 
        self.p_element_sum = props.get('p_element_sum', 0)  # P element before being diveded by N MOVING TO INTEGRATOR
        self.p_element = props.get('p_element', 0)          # Total P element value MOVING TO INTEGRATOR
        self.p_emittance = props.get('p_emittance', 0)      # P from emitter

        # Shutter properties
        self.shutter_open_ = props.get('shutter_open',0)
        self.shutter_open_time_ = props.get('shutter_open_time', 0.1)

        # Sampler 
        self.sampler_ = None

        # Shape and medium (mitsuba will figure this out)
        self.shape = None


    ### Evaluation functions ###

    def eval(self, si, active=True):
        #Given a ray-surface intersection, return the emitted radiance or importance traveling along the reverse direction
        radiance =mi.Color3f(0.0)
        return radiance

    def eval_direction(self, it, ds, active=True):
        #Re-evaluate the incident direct radiance/importance of the sample_direction() method.
        radiance = mi.Color3f(0.0)
        return radiance
    

    
    ### PDF Functions
    
    def pdf_direction(self, it, ds, active=True):
        # Evaluate the probability density of the direct sampling method implemented by the sample_direction() method.
        pdf_dir = dr.llvm.ad.Float(0.0)
        return pdf_dir
    
    def pdf_position(self, ps, active=True):
        # Evaluate the probability density of the position sampling method implemented by sample_position().
        pdf_pos = dr.llvm.ad.Float(0.0)
        return pdf_pos


    ### Sample Functions
    def sample_ray(self, time, sample1, sample2, sample3, active=True):
        # Importance sample a ray proportional to the endpoint’s sensitivity/emission profile.
        spl_ray = (mi.Ray3f, mi.Color3f)
        return spl_ray

    def sample_direction(Self, it, sample, active=True):
        # Given a reference point in the scene, sample a direction from the reference point towards the endpoint (ideally proportional to the emission/sensitivity profile)
        spl_dir = (mi.DirectionSample3f, mi.Color3f)
        return spl_dir

    def sample_position(self, time, sample, active=True):
        # Importance sample the spatial component of the emission or importance profile of the endpoint.
        spl_pos = (mi.PositionSample3f, dr.llvm.ad.Float)
        return spl_pos
    
    def sample_ray_differential(self, time, sample1, sample2, sample3, active=True):
        # Importance sample a ray differential proportional to the sensor’s sensitivity profile. Not Implimented
        spl_ray_d = (mi.RayDifferential3f, mi.Color3f)
        return spl_ray_d
    
    def sample_wavelengths(self, si, sample, active=True):
        # Importance sample a set of wavelengths according to the endpoint’s sensitivity/emission spectrum. Not Implimented
        spl_wave = (mi.Color0f(0), mi.Color3f(0))
        return spl_wave
    



    ### Getter Functions ###

    def sampler():
        return self.sampler
    
    def shutter_open():
        return self.shutter_open_
    
    def shutter_open_time():
        return self.shutter_open_time_
    
    def film():
        return self.film
    

    def get_shape():
        return self.shape
    
    def needs_aperture_sample():
        # Does the sampling technique require a sample for the aperture position?
        return True