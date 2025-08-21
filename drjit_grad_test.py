import drjit as dr
import mitsuba as mi
mi.set_variant("llvm_ad_mono")  # or "cuda_ad_mono" if you're on GPU

# Dr.Jit arrays
x_dr = mi.Float([4, 4, 4, 4])
n_dr = mi.UInt32([0, 1, 2, 3])

# Enable AD tracking on x, not on n (we differentiate wrt x)
dr.enable_grad(x_dr)

# Make float exponents: 2.0**n
exp_dr = dr.power(2,n_dr)
y_dr = dr.power(x_dr, exp_dr)
print("y_dr:", y_dr)

# Backprop sum(y) and read grad wrt x
dr.backward(dr.sum(y_dr))
print("grad(x_dr) (Dr.Jit):", dr.grad(x_dr))

# (optional) clear grad if reusing:
dr.set_grad(x_dr, mi.Float(0))
dr.disable_grad(x_dr)
