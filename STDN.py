import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class CropLayer(nn.Module):

    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)

        center_offset_from_origin_border = 0
        ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
        hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)

        self.ver_conv_crop_layer = nn.Identity()
        ver_conv_padding = ver_pad_or_crop
        self.hor_conv_crop_layer = nn.Identity()
        hor_conv_padding = hor_pad_or_crop

        self.ver_conv = nn.Conv2d(inplanes, planes, kernel_size=(3, 1), padding=ver_conv_padding, stride=stride,
                                  bias=False)
        self.hor_conv = nn.Conv2d(inplanes, planes, kernel_size=(1, 3), padding=hor_conv_padding, stride=stride,
                                  bias=False)
        self.ver_bn = nn.BatchNorm2d(planes)
        self.hor_bn = nn.BatchNorm2d(planes)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        ver_out1 = self.ver_conv_crop_layer(x)
        ver_out1 = self.ver_conv(ver_out1)
        ver_out1 = self.ver_bn(ver_out1)

        hor_out1 = self.hor_conv_crop_layer(x)
        hor_out1 = self.hor_conv(hor_out1)
        hor_out1 = self.hor_bn(hor_out1)

        out = out + ver_out1 + hor_out1

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class STDN(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(STDN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def stdn18(**kwargs):
    """Constructs a STDN-18 model.    """

    model = STDN(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def stdn34(**kwargs):
    """Constructs a STDN-34 model.    """

    model = STDN(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


