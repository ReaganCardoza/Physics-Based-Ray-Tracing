import mitsuba as mi
import numpy as np
import drjit as dr

#mi.set_variant("llvm_ad_rgb")

class UltraBSDF(mi.BSDF):
    def __init__(self, props):
        super().__init__(props)

        # Get accoustic properties
        self.impedance = mi.Float(1.54)
        if props.has_property('impedance'):
            self.impedance = mi.Float(props['impedance'])

        self.roughness = mi.Float(0.5)
        if props.has_property('roughness'):
            self.roughness = mi.Float(props['roughness'])

        self.pdf_max = mi.Float(0.2)
        if props.has_property('pdf_max'):
            self.pdf_max = mi.Float(props['pdf_max'])
        
        # Set Appropriate flags
        reflection_flags = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags


    
    def _ggx_sample(self, wi_world, n_world, sample_2d: mi.Point2f):
        frame = mi.Frame3f(n_world)
        wi = frame.to_local(wi_world)
        alpha = self.roughness

        wi_stretched = mi.Vector3f(alpha * wi.x, alpha * wi.y, wi.z)
        wi_stretched = dr.normalize(wi_stretched)

        inv_len = dr.rsqrt(dr.maximum(1.0 - wi_stretched.z * wi_stretched.z, 1e-7))
        T1 = mi.Vector3f(wi_stretched.y * inv_len, -wi_stretched.x * inv_len, 0.0)
        T2 = dr.cross(wi_stretched, T1)

        # ✅ needs a 2D sample
        d = mi.warp.square_to_uniform_disk_concentric(sample_2d)

        S = 0.5 * (1.0 + wi_stretched.z)
        d_y_sq = dr.maximum(1.0 - d.x * d.x, 0.0)
        d.y = (1.0 - S) * dr.sqrt(d_y_sq) + S * d.y

        m_stretched = d.x * T1 + d.y * T2 + dr.sqrt(dr.maximum(1.0 - d.x * d.x - d.y * d.y, 0.0)) * wi_stretched
        m = mi.Vector3f(alpha * m_stretched.x, alpha * m_stretched.y, m_stretched.z)
        return dr.normalize(m)

    

    def ggx_pdf(self, cos_theta, alpha, pdf_max):
        # cos_theta in [-1, 1]
        cos_theta = dr.clamp(cos_theta, -1.0, 1.0)
        sin_theta = dr.sqrt(dr.maximum(1.0 - cos_theta * cos_theta, 0.0))
        a2 = alpha * alpha
        # No weird Int casts; keep it Float
        denom = (a2 - 1.0) * (cos_theta * cos_theta) + 1.0
        D = a2 / (dr.pi * denom * denom)     # NDF
        pdf = D * sin_theta                  # convert to spherical measure
        return pdf / dr.maximum(pdf_max, 1e-6)

    


    def sample(self, ctx, si, sample1, sample2, pdf_max, active = True):
        
        # Directions and angles
        incident_direction = si.wi
        surface_normal = si.sh_frame.n


        # Sample Micro facet normals
        m = self._ggx_sample(si.wi, si.sh_frame.n, sample1)



        # Ensure proper orientation
        m = dr.select(dr.dot(m, incident_direction) < 0, m, -m)
        cos_wi_m = dr.dot(incident_direction, m) 

        # Snells ratio calculations
        entering = dr.dot(m, incident_direction) > 0
        medium_z = 1.2
        Z1 = dr.select(entering, medium_z, self.impedance)
        Z2 = dr.select(entering, self.impedance, medium_z)

        

        snells_ratio = mi.Float(Z1 / Z2)





        # Amplitude calculations
        
        cosTr = dr.abs(dr.dot(m, incident_direction))
        sqrt_arg = 1 - (snells_ratio ** 2) * (1 - cosTr**2)
        cosTt = dr.sqrt(dr.maximum(sqrt_arg, 0.0))
        denom = Z1 * cosTr + Z2 * cosTt
        Ar = (Z1 * cosTr - Z2 * cosTt) / denom
        At = 1. - Ar





        reflected_direction = incident_direction + 2 * cos_wi_m * m
        transmission_direction = snells_ratio * reflected_direction + (snells_ratio * cosTr - cosTt) * m

        reflected_direction = mi.Vector3f(reflected_direction)
        transmission_direction = mi.Vector3f(transmission_direction)


        tir = sqrt_arg < 0

        

        # If Ar is amplitude, square it for energy
        prob_reflect = Ar * Ar
        # Force reflection on TIR, otherwise use Russian roulette
        prob_reflect_bool = (sample2 < prob_reflect)[0]
        select_reflect = dr.select(tir, True, prob_reflect_bool)

        chosen_dir = dr.select(select_reflect, reflected_direction,
                                            transmission_direction)
        

        
        # Outgoing direction PDF
        pdf_m = self.ggx_pdf(cos_wi_m, self.roughness, self.pdf_max)
        pdf_reflect = pdf_m / (4 * dr.abs(cos_wi_m))
        cos_wo_m = dr.dot(transmission_direction, m)
        abs_n_wi = dr.abs(dr.dot(surface_normal, incident_direction))
        abs_n_wo = dr.maximum(dr.abs(dr.dot(surface_normal, transmission_direction)), 1e-7)
        pdf_trans = pdf_m * snells_ratio ** 2 * dr.abs(cos_wo_m) / (abs_n_wi * abs_n_wo)

        bs = mi.BSDFSample3f()
        bs.sampled_type = dr.select(select_reflect,
                                   mi.UInt32(+mi.BSDFFlags.GlossyReflection),
                                   mi.UInt32(+mi.BSDFFlags.GlossyTransmission))
        
        bs.wo  = si.to_local(chosen_dir)
        bs.pdf = dr.select(select_reflect, pdf_reflect, pdf_trans)
        bs.eta = 1.0
        bs.sampled_component = dr.select(select_reflect, mi.UInt32(0), mi.UInt32(1))

        acoustic_response_amp = dr.select(select_reflect, Ar, At)

        acoustic_response = acoustic_response_amp 
    

        return (bs, acoustic_response)

    def eval(self, ctx, si, wo, active):
        return 0.0
    
    def pdf(self, ctx, si, wo, active):
        return 0.0
    
    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0
    
    def traverse(self, callback):
        callback.put('impedance', self.impedance, mi.ParamFlags.Differentiable)
        callback.put('roughness', self.roughness, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass




