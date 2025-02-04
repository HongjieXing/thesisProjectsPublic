import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import weight_norm

# 创建基础卷积层
def create_layer(in_channels, out_channels, kernel_size, wn=True, bn=True,
                 activation=nn.ReLU, convolution=nn.Conv2D):
    assert kernel_size % 2 == 1
    layer = []
    conv = convolution(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    if wn:
        conv = weight_norm(conv)
    layer.append(conv)
    if bn:
        layer.append(nn.BatchNorm2D(out_channels))
    if activation is not None:
        layer.append(activation())
    
    return nn.Sequential(*layer)

# 创建Encoder中的单个块
def create_encoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                         activation=nn.ReLU, layers=2):
    encoder = []
    for i in range(layers):
        # print(i)
        _in = out_channels
        _out = out_channels
        if i == 0:
            _in = in_channels
        encoder.append(create_layer(_in, _out, kernel_size, wn, bn, activation, nn.Conv2D))
    return nn.Sequential(*encoder)

# 创建Decoder中的单个块
def create_decoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                         activation=nn.ReLU, layers=2, final_layer=False):
    decoder = []
    for i in range(layers):
        _in = in_channels
        _out = in_channels
        _bn = bn
        _activation = activation
        if i == 0:
            _in = in_channels * 2
        if i == layers - 1:
            _out = out_channels
            if final_layer:
                # kernel_size = 1
                _bn = False
                _activation = None
        decoder.append(create_layer(_in, _out, kernel_size, wn, _bn, _activation, nn.Conv2DTranspose))
    return nn.Sequential(*decoder)

# 创建Encoder
def create_encoder(in_channels, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2):
    encoder = []
    for i in range(len(filters)):        
        if i == 0:
            layers = 2
        elif i == (len(filters) - 1):
            kernel_size = 1
            layers = 3
        else:
            layers = 2 # ! ! !
            
        if i == 0:
            encoder_layer = create_encoder_block(in_channels, filters[i], kernel_size, wn, bn, activation, layers)
        else:
            encoder_layer = create_encoder_block(filters[i - 1], filters[i], kernel_size, wn, bn, activation, layers)
        encoder = encoder + [encoder_layer]
    return nn.Sequential(*encoder)

# 创建Decoder
def create_decoder(out_channels, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2):
    decoder = []
    for i in range(len(filters)):
        if i == 0:
            layers = 2
        elif i == (len(filters) - 1):
            kernel_size = 1
            layers = 3
        else:
            layers = 2 # ! ! !
        
        if i == 0:
            decoder_layer = create_decoder_block(filters[i], out_channels, kernel_size, wn, bn, activation, layers,
                                                 final_layer=True)
        else:
            decoder_layer = create_decoder_block(filters[i], filters[i - 1], kernel_size, wn, bn, activation, layers,
                                                 final_layer=False)
        decoder = [decoder_layer] + decoder
    return nn.Sequential(*decoder)

# 创建DeepCFD网络
class UNetEx(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64], layers=2,
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None): 
        super().__init__()
        assert len(filters) > 0
        self.final_activation = final_activation
        self.encoder = create_encoder(in_channels, filters, kernel_size, weight_norm, batch_norm, activation, layers)
        decoders = []
        for i in range(out_channels):
            decoders.append(create_decoder(1, filters, kernel_size, weight_norm, batch_norm, activation, layers))
        self.decoders = nn.Sequential(*decoders)

    def encode(self, x):
        tensors = []
        indices = []
        sizes = []
        # for encoder in self.encoder:
        #     x = encoder(x)
        #     sizes.append(x.shape)
        #     tensors.append(x)
        #     x, ind = F.max_pool2d(x, 2, 2, return_mask=True)
        #     indices.append(ind)
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)
            sizes.append(x.shape)
            tensors.append(x)
            if i != len(self.encoder) - 1:                
                x, ind = F.max_pool2d(x, 2, 2, return_mask=True)
                indices.append(ind)
        
        return x, tensors, indices, sizes

    def decode(self, _x, _tensors, _indices, _sizes):
        y = []
        for _decoder in self.decoders:
            x = _x
            tensors = _tensors[:]
            indices = _indices[:]
            sizes = _sizes[:]
            # for decoder in _decoder:
            #     tensor = tensors.pop()
            #     size = sizes.pop()
            #     ind = indices.pop()
            #     # 反池化操作，为上采样
            #     x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
            #     x = paddle.concat([tensor, x], axis=1)
            #     x = decoder(x)
            for i, decoder in enumerate(_decoder):
                tensor = tensors.pop()
                size = sizes.pop()
                # 反池化操作，为上采样
                if i == 0:
                    x = tensor
                else :
                    ind = indices.pop()
                    x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
                x = paddle.concat([tensor, x], axis=1)
                x = decoder(x)

            y.append(x)
        return paddle.concat(y, axis=1)

    def forward(self, x):
        x, tensors, indices, sizes = self.encode(x)
        x = self.decode(x, tensors, indices, sizes)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x