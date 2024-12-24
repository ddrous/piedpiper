import jax
import jax.numpy as jnp
from jax import jit, pmap, vmap
import time
import numpy as np

from functools import partial

class JAXStream:
    def __init__(self, array_size, devices=None):
        self.array_size = array_size

    def init_arrays(self, initA, initB, initC):
        self.a = initA * jnp.ones(self.array_size, dtype=jnp.float32)
        self.b = initB * jnp.ones(self.array_size, dtype=jnp.float32)
        self.c = initC * jnp.ones(self.array_size, dtype=jnp.float32)


    @staticmethod
    @jit
    # @partial(vmap, in_axes=(None, 0))
    def _mul(scalar, c):
        return scalar * c

    @staticmethod
    @jit
    def _add(a, b):
        return a + b

    # @staticmethod
    # @pmap
    # def _triad(scalar, b, c):
    #     return b + scalar * c


    @staticmethod
    @jit
    # @partial(vmap, in_axes=(None, 0, 0))
    def _triad(scalar, b, c):
        # print("\nI am being traced\n")
        # print("Jax shape:", b.shape)
        return b + scalar * c


    @staticmethod
    def _nstream(scalar, a, b, c):
        return a + b + scalar * c

    @staticmethod
    @jit
    def _copy(a):
        return jnp.array(a)

    # @staticmethod
    # @jit
    # def _triad_jit(scalar, b, c):
    #     return b + scalar * c

    def copy(self):
        self.b = JAXStream._copy(self.a)
        _ = jnp.sum(self.b).block_until_ready()

    def mul(self, scalar):
        self.b = self._mul(scalar, self.c).block_until_ready()

    def add(self):
        self.c = self._add(self.a, self.b).block_until_ready()

    def triad(self, scalar):
        self.a = self._triad(scalar, self.b, self.c).block_until_ready()

    def nstream(self, scalar):
        self.a = self._nstream(scalar, self.a, self.b, self.c)

    def dot(self):
        return jnp.sum(self.a * self.b, axis=0).block_until_ready()

    def time_copy(self):
        start_time = time.time()
        self.copy()
        end_time = time.time()
        return end_time - start_time

    def time_mul(self, scalar):
        start_time = time.time()
        self.mul(scalar)
        end_time = time.time()
        return end_time - start_time

    def time_add(self):
        start_time = time.time()
        self.add()
        end_time = time.time()
        return end_time - start_time

    # @jit
    def time_triad(self, scalar):
        start_time = time.time()
        # self.triad(scalar)
        self.a = self._triad(scalar, self.b, self.c).block_until_ready()
        end_time = time.time()
        return end_time - start_time

    def time_nstream(self, scalar):
        start_time = time.time()
        self.nstream(scalar)
        end_time = time.time()
        return end_time - start_time
    
    def time_dot(self):
        start_time = time.time()
        self.dot()
        end_time = time.time()
        return end_time - start_time

    def benchmark(self, num_iterations=100):
        current_time = time.strftime('%a %d %b %H:%M:%S %Z %Y', time.localtime())
        print(current_time)
        print("BabelStream")
        print("Version: 3.4")  
        print("Implementation: JAX")

        print("Jax devices available:", jax.devices() )

        # print("Jaxpr for triad:", jax.make_jaxpr(self._triad)(0.4, self.b, self.c))

        print(f"Running kernels {num_iterations} times")
        print("Precision: double")
        array_size_mbytes = self.array_size * 4 / (1024 * 1024)  # Assuming float64 (8 bytes per item)
        total_size_mbytes = 3 * array_size_mbytes  
        print(f"Array size: {array_size_mbytes:.1f} MB (={array_size_mbytes/1024:.1f} GB)")
        print(f"Total size: {total_size_mbytes:.1f} MB (={total_size_mbytes/1024:.1f} GB)")
        print("Function    GBytes/sec  Min (sec)   Max         Average")
    
        scalar = jnp.array([0.4], dtype=jnp.float32)
    

        def print_timings(func, label, *args):
            func(*args)
            timings = np.zeros((num_iterations))
            for i in range(num_iterations):
                timings[i] = func(*args)
            avg_time = sum(timings) / len(timings)
            bandwidth = array_size_mbytes / (1024*min(timings[1:]))
            print(f"{label:<10} {bandwidth:>10.2f}  {min(timings):>10.5f}     {max(timings):>10.5f}     {avg_time:>10.5f}")

        def run_time(func, *args):
            start_time = time.time()
            func(*args)
            end_time = time.time()
            return end_time - start_time

        print_timings(run_time, "Copy", self.copy)
        print_timings(run_time, "Mul", self.mul, scalar)
        print_timings(run_time, "Add", self.add)
        print_timings(run_time, "Triad", self.triad, scalar)
        print_timings(run_time, "Dot", self.dot)


# with jax.default_device(jax.devices("gpu")[0]):
    # stream = JAXStream(33554432, devices=[1])
stream = JAXStream(600003840, devices=[1])      ## Divisible by 1024
stream.init_arrays(0.1, 0.2, 0.3)
stream.benchmark()
