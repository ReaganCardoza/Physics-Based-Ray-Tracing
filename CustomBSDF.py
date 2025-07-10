import mitsuba as mi
import drjit as dr

#mi.set_variant("cuda_ad_mono")

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

        
        # Set Appropriate flags
        reflection_flags = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        self.m_components = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags


    
    def _ggx_sample(self, wi_world, n_world, sample):
        # Local incident direction
        frame = mi.Frame3f(n_world)
        wi = frame.to_local(wi_world)
        alpha = self.roughness

        # Stretch view vector
        wi_stretched = mi.Vector3f(alpha * wi.x, alpha * wi.y, wi.z)
        wi_stretched = dr.normalize(wi_stretched)

        # Orthoginal basis around wi stretched
        inv_len = dr.rsqrt(dr.maximum(1.0 - wi_stretched.z * wi_stretched.z, 1e-7))
        T1 = mi.Vector3f(wi_stretched.y * inv_len,
                         -wi_stretched.x * inv_len,
                         0.0)
        T2 = dr.cross(wi_stretched, T1)

        # Sample point on unit disk 
        d = mi.warp.square_to_uniform_disk_concentric(sample)

        # Stretching compensation
        S = 0.5 * (1.0 + wi_stretched.z)
        d.y = (1.0 - S) * dr.sqrt(dr.maximum(1.0 - d.x * d.x, 0.0)) + S * d.y

        # Convert slopes to normal, then unstretch
        m_stretched = d.x * T1 + d.y * T2 + dr.sqrt(dr.maximum(1.0 - d.x * d.x - d.y * d.y, 0.0)) * wi_stretched
        m = mi.Vector3f(alpha * m_stretched.x,
                        alpha * m_stretched.y,
                        dr.maximum(m_stretched.z, 0.0))
        m = dr.normalize(m)

        # GGX NDF
        cos_theta_m = m.z
        alpha2 = alpha * alpha
        denom = dr.maximum(cos_theta_m * cos_theta_m * ( alpha2 - 1.0) + 1.0, 1e-7)
        D = alpha2 / (dr.pi * denom * denom)

        # G1 for the incident direction
        tan2_theta = dr.maximum(1.0 - wi.z*wi.z, 0.0) / dr.maximum(wi.z * wi.z, 1e-7)
        G1_wi = 2.0/ (1.0 + dr.sqrt(1.0 + alpha2 * tan2_theta))

        # Visable-normal pdf
        pdf_m = G1_wi * dr.abs(dr.dot(wi, m)) * D / dr.maximum(wi.z, 1e-7)

        return m, D, G1_wi, pdf_m
    


    def sample(self, ctx, si, sample1, sample2, active = True):
        
        # Directions and angles
        incident_direction = si.wi
        surface_normal = si.sh_frame.n


        # Sample Micro facet normals
        m, D, G1_wi, pdf_m = self._ggx_sample(si.wi, si.n, sample2)
        cos_wi_m = dr.dot(incident_direction, m) 

        # Snells ratio calculations
        entering = dr.dot(m, incident_direction) > 0
        medium_z = 1.54
        Z1 = dr.select(entering, medium_z, self.impedance)
        Z2 = dr.select(entering, self.impedance, medium_z)

        

        snells_ratio = mi.Float(Z1 / Z2)

        # Directions wrt the micro facet normal
        reflected_direction = incident_direction - 2 * cos_wi_m * m
        cos_theta_i = dr.dot(incident_direction, m)
        transmission_direction = mi.refract(incident_direction, mi.Normal3f(m), cos_theta_i, snells_ratio)
        tir = dr.all(transmission_direction == 0)


        # Amplitude calculations
        cosTr = dr.dot(m, incident_direction)
        sqrt_arg = 1 - (snells_ratio ** 2) * (1 - cosTr**2)
        cosTt = dr.sqrt(dr.maximum(sqrt_arg, 0.0))
        cosTt = dr.select(cosTr > 0, cosTt, -cosTt)
        denom = dr.maximum(Z1 * cosTt + Z2 * cosTr, 1e-8)
        Ar = -(Z1 * cosTr - Z2 * cosTt) / denom
        At = mi.Float(1. - Ar)

        dr.print(Ar)

   

        

        # Russian roulette 
        prob_reflect = dr.clamp(dr.abs(Ar), 0.0, 1.0)
        select_reflect = dr.select(tir, True, sample1 < prob_reflect)

        chosen_dir = dr.select(select_reflect, reflected_direction,
                                            transmission_direction)
        
        # Outgoing direction PDF
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

        acoustic_response = dr.select(select_reflect, Ar, At)
    

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




