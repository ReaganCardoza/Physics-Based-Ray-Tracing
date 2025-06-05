# TO RUN: python /home/river/USM/USMain.py

import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

# Import Custom Classes
from CustomEmmitter import UltraRayEmitter
from CustomSensor import UltraSensor

mi.set_variant("cuda_ad_rgb")

mi.register_emitter("UltraRayEmitter", lambda props: UltraRayEmitter(props))
mi.register_sensor("UltraSensor", lambda props: UltraSensor(props))

USEmitter = mi.load_dict({
    'type':'UltraRayEmitter'
    })

USSensor = mi.load_dict({
    'type':'UltraSensor'
    })

scene = mi.load_dict({
    'type' : 'scene',
    'integrator': {
        'type': 'path'
    },
    'light': {
        'type': 'constant',
        'radiance': 0.99,
    },
    'sphere' : {
        'type' : 'sphere',
    },
    'sensor': USSensor,
    }
) 

params = mi.traverse(scene)
print(params)

#image = mi.render(scene)
