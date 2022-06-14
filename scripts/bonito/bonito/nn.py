"""
Bonito nn modules.
"""

import math
import pdb
import sys
import torch
from torch import nn
from torch.nn import Module
from torch.nn.init import orthogonal_

# Our module!
sys.path.append("../../core")
from MarkonvCore import *


layers = {}


def register(layer):
    layer.name = layer.__name__.lower()
    layers[layer.name] = layer
    return layer


register(torch.nn.ReLU)
register(torch.nn.Tanh)
register(torch.nn.Identity)

@register
class Swish(torch.nn.SiLU):
    pass


@register
class Serial(torch.nn.Sequential):

    def __init__(self, sublayers):
        super().__init__(*sublayers)

    def forward(self, x, return_features=False):
        if return_features:
            fmaps = []
            for layer in self:
                x = layer(x)
                fmaps.append(x)
            return x, fmaps
        return super().forward(x)

    def to_dict(self, include_weights=False):
        return {
            'sublayers': [to_dict(layer, include_weights) for layer in self._modules.values()]
        }


@register
class Reverse(Module):

    def __init__(self, sublayers):
        super().__init__()
        self.layer = Serial(sublayers) if isinstance(sublayers, list) else sublayers

    def forward(self, x):
        return self.layer(x.flip(0)).flip(0)

    def to_dict(self, include_weights=False):
        if isinstance(self.layer, Serial):
            return self.layer.to_dict(include_weights)
        else:
            return {'sublayers': to_dict(self.layer, include_weights)}


@register
class Convolution(Module):
    def __init__(self, insize, size, winlen, stride=1, padding=0, bias=True, activation=None, bn=False, residue=False):
        super().__init__()
        self.conv = torch.nn.Conv1d(insize, size, winlen, stride=stride, padding=padding, bias=bias)
        self.bn = bn
        self.residue = residue
        if bn:
            self.batchNorm = torch.nn.BatchNorm1d(size)
        if self.residue:
            sys.stderr.write("> Using residue for Convolution.\n")
        self.activation = layers.get(activation, lambda: activation)()

    def forward(self, x):
        if self.residue:
            residual = x
        x = self.conv(x)
        if self.bn:
            x = self.batchNorm(x)
        if self.residue:
            x += residual
        if self.activation is not None:
            return self.activation(x)
        return x

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.conv.in_channels,
            "size": self.conv.out_channels,
            "bias": self.conv.bias is not None,
            "winlen": self.conv.kernel_size[0],
            "stride": self.conv.stride[0],
            "padding": self.conv.padding[0],
            "activation": self.activation.name if self.activation else None,
            "bn": self.bn,
            "residue": self.residue
        }
        if include_weights:
            raise NotImplementedError
            res['params'] = {
                'W': self.conv.weight, 'b': self.conv.bias if self.conv.bias is not None else []
            }
        return res

@register
class Transformer(Module):
    def __init__(self, insize, size=None, bias=False, reverse=False, activation="gelu"):
        super().__init__()
        self.insize = insize
        self.activation = activation
        if activation == "swish":
            activation = torch.nn.functional.silu
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=insize, nhead=insize//64, dim_feedforward=4*insize, activation=activation)

    def forward(self, x):
        return self.encoder_layer(x)

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.insize,
        }
        if include_weights:
            raise NotImplementedError
        return res


@register
class LiteTransformer(Module):
    def __init__(self, insize, size=None, bias=False, reverse=False, num_layers=6, kernel_size="default", **kwargs):
        super().__init__()
        self.insize = insize
        self.num_layers = num_layers

        from bonito.litetr.modules import EncoderLayer
        self.stack_layers = nn.ModuleList(
            [EncoderLayer(index=i, d_model=insize, d_ff=insize, n_head=insize//64, dropout=0.1, kernel_size=kernel_size) for i in range(
                num_layers)])
        self.layer_norm = nn.LayerNorm(insize, eps=1e-6)

    def forward(self, x):
        # NTC
        src_mask = torch.zeros(x.size(0),1,x.size(1),dtype=torch.uint8).to(x.device)

        for layer in self.stack_layers:
            x, enc_slf_attn = layer(x, src_mask)
        x = self.layer_norm(x)
        return x

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.insize,
            "num_layers": self.num_layers
        }
        if include_weights:
            raise NotImplementedError
        return res


@register
class BertTransformer(Module):
    def __init__(self, insize, size=None, bias=False, reverse=False, **kwargs):
        super().__init__()
        from bonito.transformer import TransformerBlock

        self.insize = insize
        self.encoder_layer = TransformerBlock(insize, insize//64, insize*4, 0.1)

    def forward(self, x):
        x = self.encoder_layer(x, None)
        return x

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.insize,
        }
        if include_weights:
            raise NotImplementedError
        return res


@register
class PretrainedTransformer(Module):
    def __init__(self, insize, size=None, bias=False, reverse=False, **kwargs):#, activation="gelu"):
        super().__init__()
        from transformers import BertConfig, BertForPreTraining

        config = BertConfig.from_json_file("/home/tanyh/Markonv_OP_PyTorch-dev/pretrained/bert_config.json")
        bert = BertForPreTraining.from_pretrained("/home/tanyh/Markonv_OP_PyTorch-dev/pretrained/bert_model.ckpt.index", from_tf=True, config=config)

        self.encoder_layers = bert.bert.encoder.layer

        #self.insize = insize
        #self.activation = activation
        #if activation == "swish":
        #    activation = torch.nn.functional.silu
        #self.encoder_layer = TransformerBlock(insize, insize//64, insize*4, 0.1)#torch.nn.TransformerEncoderLayer(d_model=insize, nhead=insize//64, dim_feedforward=4*insize, activation=activation)

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)[0]
        return x

    def to_dict(self, include_weights=False):
        res = {
            #"insize": self.insize,
        }
        if include_weights:
            raise NotImplementedError
        return res


@register
class TransformerR(Module):
    def __init__(self, insize, size=None, bias=False, reverse=False, activation="gelu"):
        super().__init__()
        self.insize = insize
        self.reverse = reverse
        self.activation = activation
        if activation == "swish":
            activation = torch.nn.functional.silu
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=insize, nhead=insize//64, dim_feedforward=4*insize, activation=activation)

    def forward(self, x):
        if self.reverse: x = x.flip(0)
        y = self.encoder_layer(x)
        if self.reverse: y = y.flip(0)
        return y

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.insize,
            "reverse": self.reverse
        }
        if include_weights:
            raise NotImplementedError
        return res


@register
class ResNet(Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


@register
class Embedding(Module):
    def __init__(self, sizes, batch_norm=True, last_activation=False, activation=None, last_batchnorm = False):
        super(Embedding, self).__init__()
        self.sizes = sizes
        self.batch_norm = batch_norm
        self.last_activation = last_activation
        self.last_batchnorm = last_batchnorm
        self.activation = activation
        activationLayer = layers.get(activation, lambda: activation)()
        mlplayers = []
        for s in range(len(sizes) - 1):
            mlplayers += [
                torch.nn.Conv1d(sizes[s], sizes[s + 1], kernel_size=1),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm else None,
                activationLayer
            ]
        mlplayers = [l for l in mlplayers if l is not None]
        if batch_norm and (not last_batchnorm):
            mlplayers = mlplayers[:-2]+mlplayers[-1:]
        if not self.last_activation:
            mlplayers = mlplayers[:-1]

        self.network = Serial(mlplayers)

    def forward(self, x):
        return self.network(x)

    def to_dict(self, include_weights=False):
        res = {
            "sizes": self.sizes,
            "batch_norm": self.batch_norm,
            "last_activation":self.last_activation,
            "activation": self.activation,
            "last_batchnorm": self.last_batchnorm
        }
        if include_weights:
            raise NotImplementedError
        return res


@register
class AddPositionalEmbedding(Module):
    """ From 
    https://github.com/codertimo/BERT-pytorch/issues/53
    """
    
    def __init__(self, d_model, max_len=2001, learnable=False, rnn_type="transformer"):
        super().__init__()
        self.rnn_type = rnn_type
        if "transformer" in rnn_type:
            sys.stderr.write("> Add positional embedding.\n")
            self.d_model = d_model
            self.max_len = max_len
            self.learnable = learnable
            if self.learnable:
                z = torch.arange(end=max_len)
                z.require_grad = False
                self.register_buffer('z', z)

                self.embedding = torch.nn.Embedding(max_len,d_model)
            else:
                # Compute the positional encodings once in log space.
                pe = torch.zeros(max_len, d_model).float()
                pe.require_grad = False

                position = torch.arange(0, max_len).float().unsqueeze(1)
                div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)

                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)
        else:
            sys.stderr.write("> Do not add positional embedding.\n")


    def forward(self, x):
        # NCT: bs, channel, len
        if self.rnn_type == "transformer":
            assert self.max_len >= x.size(2), "max_len should be greater than length of x."
            if self.learnable:
                return x + self.embedding(self.z[:x.size(2)]).permute((1,0))
            else:
                # pe: bs, len, channel
                return x + self.pe[:, :x.size(2)].permute((0,2,1))
        else:
            return x

    def to_dict(self, include_weights=False):
        res = {
            "rnn_type": self.rnn_type,
            "d_model": self.d_model,
            "max_len": self.max_len,
            "learnable": self.learnable
        }
        if include_weights:
            raise NotImplementedError
        return res


@register
class MarkonvLayer(Module):    
    def __init__(self, insize, size, winlen, stride=1, padding=0, bias=True, activation=None, bins=None, v=1, channel_last=False, bn=False, residue=False):
        """
        insize: in_channel
        size: markonv.kernel_number
        winlen: markonv.kernel_length
        """
        super().__init__()
        self.stride = stride
        self.padding = torch.nn.ConstantPad1d(padding, 0)
        self.bins = bins
        self.bn = bn
        self.residue = residue
        if type(winlen) == type([]):
            winlen = winlen[0]
        if bins is not None:
            raise NotImplementedError("`Bins is not None` is deprecated.")        

        if v:
            if stride != 1:
                self.markonv = MarkonvVS(int(1.5*winlen), size, channel_size=insize, channel_last=channel_last, bias=bias, stride=stride)
            else:
                self.markonv = MarkonvV(int(1.5*winlen), size, channel_size=insize, channel_last=channel_last, bias=bias)
        else:
            if stride == 1:
                self.markonv = MarkonvR(winlen, size, channel_size=insize, channel_last=channel_last, bias=bias)
            else:
                raise NotImplementedError("MarkonvR with stride != 1 has not been implemented.")

        if bn:
            self.batchNorm = torch.nn.BatchNorm1d(size)
        if self.residue:
            sys.stderr.write("> Using residue for Markonv.\n")
        self.activation = layers.get(activation, lambda: activation)()

    def forward(self, x):
        if self.residue:
            residual = x
        dtype = x.dtype
        assert x.shape[1] == self.markonv.channel_size, "x.shape[1] should equal self.markonv.channel_size"
        x = self.padding(x)
        x = self.markonv(x).to(dtype)
        if self.bn:
            x = self.batchNorm(x)
        if self.residue:
            x += residual
        if self.activation is not None:
            return self.activation(x)
        return x

    def to_dict(self, include_weights=False):
        res = {
            "stride": self.stride,
            "size": self.markonv.kernel_number,
            "winlen": self.markonv.kernel_length,
            "padding": self.padding.padding,
            "bias": self.markonv.bias is not None,
            "activation": self.activation.name if self.activation else None,
            "bn": self.bn,
            "residue": self.residue
        }
        if include_weights:
            raise NotImplementedError
            res['params'] = {
                'W': self.markonv.Kernel_Full_4DTensor, 'b': self.markonv.bias if self.markonv.bias is not None else []
            }
        return res


@register
class LinearCRFEncoder(Module):

    def __init__(self, insize, n_base, state_len, bias=True, scale=None, activation=None, blank_score=None, expand_blanks=True):
        super().__init__()
        self.scale = scale
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        self.expand_blanks = expand_blanks
        size = (n_base + 1) * n_base**state_len if blank_score is None else n_base**(state_len + 1)
        self.linear = torch.nn.Linear(insize, size, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()

    def forward(self, x):
        scores = self.linear(x)
        if self.activation is not None:
            scores = self.activation(scores)
        if self.scale is not None:
            scores = scores * self.scale
        if self.blank_score is not None and self.expand_blanks:
            T, N, C = scores.shape
            scores = torch.nn.functional.pad(
                scores.view(T, N, C // self.n_base, self.n_base),
                (1, 0, 0, 0, 0, 0, 0, 0),
                value=self.blank_score
            ).view(T, N, -1)
        return scores

    def to_dict(self, include_weights=False):
        res = {
            'insize': self.linear.in_features,
            'n_base': self.n_base,
            'state_len': self.state_len,
            'bias': self.linear.bias is not None,
            'scale': self.scale,
            'activation': self.activation.name if self.activation else None,
            'blank_score': self.blank_score,
        }
        if include_weights:
            res['params'] = {
                'W': self.linear.weight, 'b': self.linear.bias
                if self.linear.bias is not None else []
            }
        return res


@register
class Permute(Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def to_dict(self, include_weights=False):
        return {'dims': self.dims}


def truncated_normal(size, dtype=torch.float32, device=None, num_resample=5):
    x = torch.empty(size + (num_resample,), dtype=torch.float32, device=device).normal_()
    i = ((x < 2) & (x > -2)).max(-1, keepdim=True)[1]
    return torch.clamp_(x.gather(-1, i).squeeze(-1), -2, 2)


class RNNWrapper(Module):
    def __init__(
            self, rnn_type, *args, reverse=False, orthogonal_weight_init=True, disable_state_bias=True, bidirectional=False, ln=False, **kwargs
    ):
        super().__init__()
        if reverse and bidirectional:
            raise Exception("'reverse' and 'bidirectional' should not both be set to True")
        self.reverse = reverse
        self.rnn = rnn_type(*args, bidirectional=bidirectional, **kwargs)
        self.ln = ln
        self.init_orthogonal(orthogonal_weight_init)
        self.init_biases()
        if disable_state_bias: self.disable_state_bias()
        if ln:
            self.layerNorm = torch.nn.LayerNorm(self.rnn.input_size)

    def forward(self, x):
        if self.reverse: x = x.flip(0)
        y, h = self.rnn(x)
        if self.reverse: y = y.flip(0)
        if self.ln:
            y = self.layerNorm(y)
        return y

    def init_biases(self, types=('bias_ih',)):
        for name, param in self.rnn.named_parameters():
            if any(k in name for k in types):
                with torch.no_grad():
                    param.set_(0.5*truncated_normal(param.shape, dtype=param.dtype, device=param.device))

    def init_orthogonal(self, types=True):
        if not types: return
        if types == True: types = ('weight_ih', 'weight_hh')
        for name, x in self.rnn.named_parameters():
            if any(k in name for k in types):
                for i in range(0, x.size(0), self.rnn.hidden_size):
                    orthogonal_(x[i:i+self.rnn.hidden_size])

    def disable_state_bias(self):
        for name, x in self.rnn.named_parameters():
            if 'bias_hh' in name:
                x.requires_grad = False
                x.zero_()


@register
class LSTM(RNNWrapper):

    def __init__(self, size, insize, bias=True, reverse=False, activation=None, ln=False, **kwargs):
        super().__init__(torch.nn.LSTM, size, insize, bias=bias, reverse=reverse, ln=ln)

    def to_dict(self, include_weights=False):
        res = {
            'size': self.rnn.hidden_size,
            'insize': self.rnn.input_size,
            'bias': self.rnn.bias,
            'reverse': self.reverse,
            'ln': self.ln
        }
        if include_weights:
            res['params'] = {
                'iW': self.rnn.weight_ih_l0.reshape(4, self.rnn.hidden_size, self.rnn.input_size),
                'sW': self.rnn.weight_hh_l0.reshape(4, self.rnn.hidden_size, self.rnn.hidden_size),
                'b': self.rnn.bias_ih_l0.reshape(4, self.rnn.hidden_size)
            }
        return res


@register
class BiLSTM(RNNWrapper):

    def __init__(self, size, insize, bias=True, reverse=False, activation=None):
        super().__init__(torch.nn.LSTM, size, insize, bias=bias, reverse=False, bidirectional=True)

    def to_dict(self, include_weights=False):
        res = {
            'size': self.rnn.hidden_size,
            'insize': self.rnn.input_size,
            'bias': self.rnn.bias,
            'reverse': self.reverse,
        }
        if include_weights:
            raise NotImplementedError
            res['params'] = {
                'iW': self.rnn.weight_ih_l0.reshape(4, self.rnn.hidden_size, self.rnn.input_size),
                'sW': self.rnn.weight_hh_l0.reshape(4, self.rnn.hidden_size, self.rnn.hidden_size),
                'b': self.rnn.bias_ih_l0.reshape(4, self.rnn.hidden_size)
            }
        return res


def to_dict(layer, include_weights=False):
    if hasattr(layer, 'to_dict'):
        return {'type': layer.name, **layer.to_dict(include_weights)}
    return {'type': layer.name}


def from_dict(model_dict, layer_types=None):
    model_dict = model_dict.copy()
    if layer_types is None:
        layer_types = layers
    type_name = model_dict.pop('type')
    typ = layer_types[type_name]
    if 'sublayers' in model_dict:
        sublayers = model_dict['sublayers']
        model_dict['sublayers'] = [
            from_dict(x, layer_types) for x in sublayers
        ] if isinstance(sublayers, list) else from_dict(sublayers, layer_types)
    try:
        layer = typ(**model_dict)
    except Exception as e:
        raise Exception(f'Failed to build layer of type {typ} with args {model_dict}') from e
    return layer

