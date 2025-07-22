from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['meta_resnet']


# --- Start of MetaModule definitions (as provided in your first file) ---
# This part is copied directly from your initial MetaModule file.
def to_var(x, requires_grad=None, is_cuda=True):
    if is_cuda and torch.cuda.is_available():
        x = x.cuda()
    if requires_grad is not None:
        x.requires_grad_(requires_grad)
    return x

class MetaModule(nn.Module):
    # base class for meta-learning modules
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)


    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

# --- End of MetaModule definitions ---


def meta_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class MetaBasicBlock(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(MetaBasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = meta_conv3x3(inplanes, planes, stride)
        self.bn1 = MetaBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True) # ReLU doesn't have learnable params, so nn.ReLU is fine
        self.conv2 = meta_conv3x3(planes, planes)
        self.bn2 = MetaBatchNorm2d(planes)
        self.downsample = downsample # Downsample might contain MetaConv2d/MetaBatchNorm2d
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class MetaBottleneck(MetaModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(MetaBottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class MetaResNet(MetaModule):

    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10):
        super(MetaResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = MetaBasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = MetaBottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = num_filters[0]
        self.conv1 = MetaConv2d(3, num_filters[0], kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = MetaBatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8) # AvgPool2d doesn't have learnable params, so nn.AvgPool2d is fine
        self.fc = MetaLinear(num_filters[3] * block.expansion, num_classes)

        # Weight initialization needs to be adapted for MetaModule,
        # as parameters are buffers.
        # We need to access the actual tensors (weight/bias) and apply init.
        self._weights_init()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks-1)))

        return nn.Sequential(*layers)

    def _weights_init(self):
        # Iterate over all named_leaves (parameters) in the MetaModule
        for name, m in self.named_modules():
            if isinstance(m, MetaConv2d):
                # Access the weight buffer directly for initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # Bias is not False in MetaConv2d in this ResNet, but good to have
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MetaLinear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.fc)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], MetaBottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], MetaBasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x

        # The BasicBlock/Bottleneck forward returns (out, preact) if is_last=True,
        # otherwise just out.
        # We need to handle this to capture f1_pre, f2_pre, f3_pre.
        # Since layer1, layer2, layer3 are nn.Sequential, their forward only returns the last module's output.
        # So we need to ensure the last block in each layer returns preact.
        # The original code already sets `is_last` for the last block correctly.
        # We assume the `forward` method of MetaBasicBlock/MetaBottleneck handles the tuple return.

        out_layer1 = self.layer1(x)
        f1_pre = None
        if isinstance(out_layer1, tuple):
            x, f1_pre = out_layer1
        else:
            x = out_layer1
        f1 = x

        out_layer2 = self.layer2(x)
        f2_pre = None
        if isinstance(out_layer2, tuple):
            x, f2_pre = out_layer2
        else:
            x = out_layer2
        f2 = x

        out_layer3 = self.layer3(x)
        f3_pre = None
        if isinstance(out_layer3, tuple):
            x, f3_pre = out_layer3
        else:
            x = out_layer3
        f3 = x


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f4 = x
        x = self.fc(x)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4], x
            else:
                return [f0, f1, f2, f3, f4], x
        else:
            return x


def meta_resnet8(**kwargs):
    return MetaResNet(8, [16, 16, 32, 64], 'basicblock', **kwargs)


def meta_resnet14(**kwargs):
    return MetaResNet(14, [16, 16, 32, 64], 'basicblock', **kwargs)


def meta_resnet20(**kwargs): # 41.6M
    return MetaResNet(20, [16, 16, 32, 64], 'basicblock', **kwargs)


def meta_resnet32(**kwargs): # 70.4M
    return MetaResNet(32, [16, 16, 32, 64], 'basicblock', **kwargs)


def meta_resnet44(**kwargs):
    return MetaResNet(44, [16, 16, 32, 64], 'basicblock', **kwargs)


def meta_resnet56(**kwargs):
    return MetaResNet(56, [16, 16, 32, 64], 'basicblock', **kwargs)


def meta_resnet110(**kwargs):
    return MetaResNet(110, [16, 16, 32, 64], 'basicblock', **kwargs)


def meta_resnet8x4(**kwargs):
    return MetaResNet(8, [32, 64, 128, 256], 'basicblock', **kwargs)


def meta_resnet8x4_double(**kwargs):
    return MetaResNet(8, [64, 128, 256, 512], 'basicblock', **kwargs)


def meta_resnet32x4(**kwargs):
    return MetaResNet(32, [32, 64, 128, 256], 'basicblock', **kwargs)


def gap_loss(y_s, y_t, temp_stu, temp_tea):

    loss = (torch.logsumexp(y_t/temp_tea, 1)-torch.logsumexp(y_s/temp_stu, 1)).mean(0)

    return loss

if __name__ == '__main__':
    import torch
    # from thop import clever_format, profile # thop doesn't work directly with MetaModules
    # You'll need a custom way to count MACs/params for MetaModules if needed for analysis.

    model = meta_resnet32()
    print(model)

    # Example of accessing meta-learnable parameters
    print("\nMeta-learnable parameters:")
    for name, param in model.named_params():
        print(f"{name}: {param.shape}")

    input_data = torch.randn(1, 3, 32, 32)
    output = model(input_data)
    print(f"\nOutput shape: {output.shape}")

    # Example of a forward pass with feature extraction and preact
    features, output_with_feat = model(input_data, is_feat=True, preact=True)
    print(f"\nNumber of feature maps: {len(features)}")
    for i, f in enumerate(features):
        print(f"Feature {i} shape: {f.shape}")
    print(f"Output shape with features: {output_with_feat.shape}")

    # To test the update_params functionality (conceptual, not a full training loop)
    # This shows how you would generate gradients and then manually update parameters.
    dummy_loss = output.sum()
    dummy_loss.backward()

    # Now, parameters have gradients. You could iterate and update them:
    # for name, param in model.named_params():
    #     if param.grad is not None:
    #         print(f"Gradient for {name}: {param.grad.mean()}")
    # model.update_params(lr_inner=0.01) # This would update the parameters in-place
    # print("\nParameters after a dummy update (conceptual):")
    # for name, param in model.named_params():
    #     print(f"{name} (first value): {param.flatten()[0]}")