import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_mono")

class UltraBSDF(mi.BSDF):
    def __init__(self, props):
        super().__init__(props)

        # Get accoustic properties
        self.impedance = 1.54
        if props.has_property('impedance'):
            self.impedance = props['impedance']

        self.roughness = 0.5
        if props.has_property('roughness'):
            self.roughness = props['roughness']

        
        # Set Appropriate flags
        reflection_flags = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags

    def sample(self, ctx, si, sample1, sample2, active = True):


        # Snells ratio
        entering = dr.dot(si.n, si.wi) > 0
        medium_z = 1.54
        Z1 = dr.select(entering, medium_z, self.impedance)
        Z2 = dr.select(entering, self.impedance, medium_z)

        snells_ratio = Z1 / Z2

        # Directions and angles
        incident_direction = si.wi
        surface_normal = si.sh_frame.n

        cosTr = dr.dot(surface_normal, -(incident_direction))

        sqrt_arg = 1- (snells_ratio ** 2) * (1 - cosTr**2)
        sqrt_arg = dr.maximum(sqrt_arg, 0.0)
        cosTt = dr.sqrt(sqrt_arg)

        reflected_direction = incident_direction + 2 * cosTr * surface_normal
        transmission_direction = snells_ratio * incident_direction + (snells_ratio * (cosTr - cosTt)) * surface_normal

        # Amplitude calculations
        denom = Z1 * cosTt + Z2 * cosTr
        denom = dr.select(dr.abs(denom) < 1e-8, 1e-8, denom)  # Avoid division by zero
        Ar = (Z1 * cosTr - Z2 * cosTt) / denom

        At = mi.Float(1. - Ar)

        # After Fresnel calculation:
        #dr.print("Impedance ratio:", snells_ratio)
        #dr.print("Ar should be [-1,1]:", dr.clamp(Ar, -1.0, 1.0))

        # Clamp to physical range:
        Ar = dr.clamp(Ar, -1.0, 1.0)
        At = dr.clamp(At, -1.0, 1.0)


        prob_reflect = dr.clamp(dr.abs(Ar), 0.0, 1.0)
        select_reflect = sample1 < prob_reflect

        chosen_dir = dr.select(select_reflect,
                               reflected_direction,
                               transmission_direction)
        
        bs = mi.BSDFSample3f()
        
        bs.sampled_type = dr.select(select_reflect,
                                    mi.UInt32(+mi.BSDFFlags.DeltaReflection),
                                    mi.UInt32(+mi.BSDFFlags.DeltaTransmission))
        bs.sampled_component = dr.select(select_reflect, mi.UInt32(0), mi.UInt32(1))
        bs.pdf = dr.select(select_reflect, prob_reflect, 1 - prob_reflect )
        bs.wo = si.to_local(chosen_dir)
        bs.eta = 1.0
        acoustic_response = dr.select(select_reflect, Ar, At)


        dr.print(cosTr)
        dr.print(cosTt)
        

        return (bs, acoustic_response)

    def eval(self, ctx, si, wo, active):
        return 0.0
    
    def pdf(self, ctx, si, wo, active):
        return 0.0
    
    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0
    
    def traverse(self, callback):
        callback.put_parameter('impedance', self.impedance, mi.ParamFlags.Differentiable)
        callback.put_parameter('roughness', self.roughness, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass



class AcousticMedium(mi.Medium):
    def __init__(self, props):
        super.__init__(props)

        self.impedance = props.get('impedance', 1.54) 
        self.speed_of_sound = props.get('speed', 1540)

    def traverse(self, callback):
        callback.put_parameter('impedance', self.impedance, mi.ParamFlags.Differentiable)
