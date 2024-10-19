
import equinox as eqx
import jax
import jax.numpy as jnp

########### Implementation of a Vnet model ###########

class DownsamplingLayer(eqx.Module):
    layer: eqx.Module
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, *, key):
        self.layer = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding="SAME", key=key)
    
    def __call__(self, x):
        return self.layer(x)

class UpsamplingLayer(eqx.Module):
    layer: eqx.Module

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, *, key):
        self.layer = eqx.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding="SAME", key=key)

    def __call__(self, x):
        return self.layer(x)

class DoubleConv(eqx.Module):
    layer_1: eqx.Module
    layer_2: eqx.Module
    activation: callable
    norm_layer: eqx.Module
    dropout_rate: float

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=jax.nn.relu, batch_norm=False, dropout_rate=0., *, key):
        k1, k2 = jax.random.split(key, 2)

        self.layer_1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, padding='SAME', key=k1)
        self.layer_2 = eqx.nn.Conv2d(out_channels, out_channels, kernel_size, padding='SAME', key=k2)
        self.activation = activation
        if batch_norm:
            self.norm_layer = eqx.nn.BatchNorm(input_size=out_channels)
        else:
            self.norm_layer = None
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        x = self.layer_1(x)
        x = self.activation(x)
        x = self.layer_2(x)
        x = self.activation(x)
        if self.norm_layer:
            x = self.norm_layer(x)
        if self.dropout_rate > 0.:
            x = eqx.nn.Dropout(self.dropout_rate)(x)
        return x



class VNet(eqx.Module):
    input_shape: tuple
    output_shape: tuple
    levels: int
    depth: int
    kernel_size: int
    activation: callable
    final_activation: callable
    batch_norm: bool
    dropout_rate: float

    ## Learnable params
    left_doubleconvs: dict
    right_doubleconvs: dict
    downsamplings: dict
    upsamplings: dict
    final_conv: eqx.Module


    def __init__(self, input_shape, output_shape, levels=5, depth=32, kernel_size=5, activation=jax.nn.relu, final_activation=jax.nn.sigmoid, batch_norm=True, dropout_rate=0., *, key):

        l_key, r_key, d_key, u_key, f_key = jax.random.split(key, 5)

        self.input_shape = input_shape      ## C, H, W
        self.output_shape = output_shape    ## C, H, W
        self.levels = levels
        self.depth = depth                  ## Number of filters in the first layer
        self.kernel_size = kernel_size
        self.activation = activation
        self.final_activation = final_activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.left_doubleconvs = {}
        self.right_doubleconvs = {}
        self.downsamplings = {}
        self.upsamplings = {}
        self.final_conv = eqx.nn.Conv2d(depth, output_shape[0], 1, padding="SAME", key=f_key)


        ## NOTE! The convolutions are not changing the number of channels, the downsampling and upsampling layers are

        d_keys = jax.random.split(d_key, levels-1)
        l_keys = jax.random.split(l_key, levels)

        self.left_doubleconvs[0] = DoubleConv(input_shape[0], depth, kernel_size, activation, batch_norm, dropout_rate, key=l_keys[0])
        for i in range(1, levels):
            self.downsamplings[i] = DownsamplingLayer(self.depth*2**(i-1), self.depth*2**(i), key=d_keys[i-1])
            self.left_doubleconvs[i] = DoubleConv(self.depth*2**(i), self.depth*2**(i), kernel_size, activation, batch_norm, dropout_rate, key=l_keys[i])

        u_keys = jax.random.split(u_key, levels-1)
        r_keys = jax.random.split(r_key, levels-1)

        for i in range(self.levels-2, -1, -1):
            self.upsamplings[i] = UpsamplingLayer(self.depth*2**(i+1), self.depth*2**i, key=u_keys[i])
            self.right_doubleconvs[i] = DoubleConv(self.depth*2**(i), self.depth*2**i, kernel_size, activation, batch_norm, dropout_rate, key=r_keys[i])


    def __call__(self, inputs):
        left = {}
        left[0] = self.left_doubleconvs[0](inputs)
        # print("     - left[0].shape =", left[0].shape)
        for i in range(1, self.levels):
            down = self.downsamplings[i](left[i-1])
            conv = self.left_doubleconvs[i](down)
            left[i] = down + conv
            # if i<self.levels-1:
            #     print(f"     - left[{i}].shape = ", left[i].shape)

        central = left[self.levels-1]
        # print(f"     - central.shape = ", central.shape)

        right = central
        for i in range(self.levels-2, -1,-1):
            up = self.upsamplings[i](right)
            add = left[i] + up
            conv = self.right_doubleconvs[i](add)
            right = up + conv
            # print(f"     - right[{i}].shape =", right.shape)

        return self.final_activation(self.final_conv(right))
