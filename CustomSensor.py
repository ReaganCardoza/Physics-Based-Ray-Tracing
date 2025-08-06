import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant("llvm_ad_mono")

class CustomSensor(mi.Sensor):
    def __init__(self, props):
        super().__init__(props)

        #Transudcer Geometry Configuration
        self.number_of_elements = props.get("number_of_elements", 128)
        self.pitch = props.get("pitch", 0.0003)
        self.element_width = props.get("element_width", 0.00027) #Not used but could be for aperature effects
        self.element_height = props.get("element_height", 0.005)

        #Signal Acquisition Settings
        #Sample RAte is the time resolution of the acquisiton in Hz
        self.sample_rate = props.get("sample_rate", 50e6)
        self.speed_of_sound = props.get("speed_of_sound", 1540.0)

        #Buffer Settings
        #Shape of the rd buffer in number elements, time samples
        self.time_samples = props.get("time_samples", 3000)


        self.channel_buffer = np.zeros((self.number_of_elements, self.time_samples), dtype=np.float32)

    def put_data(self, ray: mi.Ray3f, amplitude: float, active=True):

        x = ray.o.x.numpy()
        x_numpy = x.item()


        #Map the x poisiton to the element index
        idx = int(np.round(x_numpy / self.pitch + self.number_of_elements / 2))

        #Compute the time of flight in seconds
        t = ray.time.numpy()
        t_numpy = t.item()

        #Convert the time of flight to sample index
        index = int(np.round(t_numpy * self.sample_rate))

        #Cosine weighting for directivity
        direction = dr.normalize(-ray.d)

        #Normal direction for a linear transducer not currently handling a convex transducer
        normal = mi.Vector3f(0.0, 0.0, 1.0)

        gain = dr.maximum(0.0, dr.dot(direction, normal))

        weighted_amplitude = amplitude * gain
        amplitude_np = weighted_amplitude.numpy()
        amp = amplitude_np.item()

        #Bounds check
        if 0 <= idx < self.number_of_elements and 0 <= index < self.time_samples:
            self.channel_buffer[idx, index] += amp

    def channel_data(self):
        return self.channel_buffer

    def clear(self):
        self.channel_buffer = np.zeros((self.number_of_elements, self.time_samples), dtype=np.float32)

    def traverse(self, callback):
        callback.put_parameters("number_of_elements", self.number_of_elements)
        callback.put_parameters("pitch", self.pitch)
        callback.put_parameters("element_width", self.element_width)
        callback.put_parameters("element_height", self.element_height)
        callback.put_parameters("sample_rate", self.sample_rate)
        callback.put_parameters("speed_of_sound", self.speed_of_sound)

    def parameters(self, keys):
        self.channel_buffer = np.zeros((self.number_of_elements, self.time_samples), dtype=np.float32)




props = mi.Properties("custom")
props["number_of_elements"] = 5
props["pitch"] = 1.0
props["sample_rate"] = 10.0
props["time_samples"] = 20  # 5 elements, 20 time samples

sensor = CustomSensor(props)

test_rays = [
    mi.Ray3f(o=[-2.0, 0, 0], d=[0, 0, -1], time=1.0),
    mi.Ray3f(o=[0.0, 0, 0],  d=[0, 0, -1], time=1.5),
    mi.Ray3f(o=[2.0, 0, 0],  d=[0, 0.8, -1], time=0.5),
    mi.Ray3f(o=[10.0, 0, 0], d=[0, 0, -1], time=1.0),
]

amplitudes = [1.0, 2.0, 1.0, 3.0]


for ray, amp in zip(test_rays, amplitudes):
    sensor.put_data(ray, amp)

buffer = sensor.channel_data()

print("\n=== Channel Buffer ===")
print(buffer)

print("\n=== Non-zero Entries ===")
nonzeros = np.argwhere(buffer > 0)
for idx, t_idx in nonzeros:
    print(f"Element {idx}, Time {t_idx}, Value: {buffer[idx, t_idx]:.4f}")

import matplotlib.pyplot as plt

plt.imshow(sensor.channel_data(), aspect="auto", cmap="plasma")
plt.xlabel("Time")
plt.ylabel("Element")
plt.colorbar()
plt.show()
