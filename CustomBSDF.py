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

        self.roughness = 0.5
        if props.has_property('roughness'):
            self.roughness = props['roughness']

        
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

        # Snells ratio
        entering = dr.dot(si.n, si.wi) > 0
        medium_z = 1.54
        Z1 = dr.select(entering, medium_z, self.impedance)
        Z2 = dr.select(entering, self.impedance, medium_z)

        snells_ratio = Z1 / Z2



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


        #dr.print(cosTr)
        #dr.print(cosTt)
        

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




