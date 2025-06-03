import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

class UltraSensor(mi.Sensor):
    def __init__(self, props):
        super().__init__(props)

        # m_film property

        # m_needs_sample_2

        # m_needs_sample_3

    def eval(self, si, active=True):
        #Given a ray-surface intersection, return the emitted radiance or importance traveling along the reverse direction
        radiance =mi.Color3f(0.0)
        return radiance

    def eval_direction(self, it, ds, active=True):
        #Re-evaluate the incident direct radiance/importance of the sample_direction() method.
        radiance = mi.Color3f(0.0)
        return radiance
    
    def film():
        # m_needs_sample_2
        return 0 
    

    def get_shape():
        # Returns the shape which the emitter is currently attached
        return mi.Shape()
    
    def needs_aperture_sample():
        # Does the sampling technique require a sample for the aperture position?
        return True
    
    def pdf_direction(self, it, ds, active=True):
        # Evaluate the probability density of the direct sampling method implemented by the sample_direction() method.
        pdf_dir = dr.llvm.ad.Float(0.0)
        return pdf_dir
    
    def pdf_position(self, ps, active=True):
        # Evaluate the probability density of the position sampling method implemented by sample_position().
        pdf_pos = dr.llvm.ad.Float(0.0)
        return pdf_pos

    def sample_direction(Self, it, sample, active=True):
        # Given a reference point in the scene, sample a direction from the reference point towards the endpoint (ideally proportional to the emission/sensitivity profile)
        spl_dir = (mi.DirectionSample3f, mi.Color3f)
        return spl_dir

    def sample_position(self, time, sample, active=True):
        # Importance sample the spatial component of the emission or importance profile of the endpoint.
        spl_pos = (mi.PositionSample3f, dr.llvm.ad.Float)
        return spl_pos
    
    def sample_ray(self, time, sample1, sample2, sample3, active=True):
        # Importance sample a ray proportional to the endpoint’s sensitivity/emission profile.
        spl_ray = (mi.Ray3f, mi.Color3f)
        return spl_ray
    
    def sample_ray_differential(self, time, sample1, sample2, sample3, active=True):
        # Importance sample a ray differential proportional to the sensor’s sensitivity profile.
        spl_ray_d = (mi.RayDifferential3f, mi.Color3f)
        return spl_ray_d
    
    def sample_wavelengths(self, si, sample, active=True):
        # Importance sample a set of wavelengths according to the endpoint’s sensitivity/emission spectrum.
        spl_wave = (mi.Color0f, mi.Color3f)
        return spl_wave
    
    def sampler():
        # Return the sensor’s sample generator
        return mi.Sampler
    
    def shutter_open()
        # Return the time value of the shutter opening event
        shutter_time = float()
        return shutter_time
    
    def shutter_open_time()
        # Return the length, for which the shutter remains open
        shutter_length = float()
        return shutter_length

    




