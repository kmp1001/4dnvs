"""PyTorch I3D architecture used by the VideoGPT FVD checkpoint.

Adapted from https://github.com/piergiaj/pytorch-i3d, matching the architecture
included with the supplied common_metrics_on_video_quality implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, size):
        if size % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        return max(self.kernel_size[dim] - (size % self.stride[dim]), 0)

    def forward(self, inputs):
        _, _, time, height, width = inputs.size()
        pad_time = self.compute_pad(0, time)
        pad_height = self.compute_pad(1, height)
        pad_width = self.compute_pad(2, width)
        padding = (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
            pad_time // 2,
            pad_time - pad_time // 2,
        )
        return super().forward(F.pad(inputs, padding))


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
    ):
        super().__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels, eps=1e-5, momentum=0.001)

    def compute_pad(self, dim, size):
        if size % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        return max(self._kernel_shape[dim] - (size % self._stride[dim]), 0)

    def forward(self, inputs):
        _, _, time, height, width = inputs.size()
        pad_time = self.compute_pad(0, time)
        pad_height = self.compute_pad(1, height)
        pad_width = self.compute_pad(2, width)
        padding = (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
            pad_time // 2,
            pad_time - pad_time // 2,
        )
        outputs = self.conv3d(F.pad(inputs, padding))
        if self._use_batch_norm:
            outputs = self.bn(outputs)
        if self._activation_fn is not None:
            outputs = self._activation_fn(outputs)
        return outputs


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()
        self.b0 = Unit3D(
            in_channels,
            out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels,
            out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            out_channels[1],
            out_channels[2],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels,
            out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            out_channels[3],
            out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(
            in_channels,
            out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, inputs):
        branch0 = self.b0(inputs)
        branch1 = self.b1b(self.b1a(inputs))
        branch2 = self.b2b(self.b2a(inputs))
        branch3 = self.b3b(self.b3a(inputs))
        return torch.cat([branch0, branch1, branch2, branch3], dim=1)


class InceptionI3d(nn.Module):
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes=400,
        spatial_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
    ):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {final_endpoint}")
        super().__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        self.end_points = {}

        endpoints = (
            (
                "Conv3d_1a_7x7",
                Unit3D(
                    in_channels,
                    64,
                    kernel_shape=[7, 7, 7],
                    stride=(2, 2, 2),
                    padding=(3, 3, 3),
                    name=name + "Conv3d_1a_7x7",
                ),
            ),
            (
                "MaxPool3d_2a_3x3",
                MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0),
            ),
            (
                "Conv3d_2b_1x1",
                Unit3D(64, 64, kernel_shape=[1, 1, 1], padding=0, name=name + "Conv3d_2b_1x1"),
            ),
            (
                "Conv3d_2c_3x3",
                Unit3D(64, 192, kernel_shape=[3, 3, 3], padding=1, name=name + "Conv3d_2c_3x3"),
            ),
            (
                "MaxPool3d_3a_3x3",
                MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0),
            ),
            ("Mixed_3b", InceptionModule(192, [64, 96, 128, 16, 32, 32], name + "Mixed_3b")),
            ("Mixed_3c", InceptionModule(256, [128, 128, 192, 32, 96, 64], name + "Mixed_3c")),
            (
                "MaxPool3d_4a_3x3",
                MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0),
            ),
            ("Mixed_4b", InceptionModule(480, [192, 96, 208, 16, 48, 64], name + "Mixed_4b")),
            ("Mixed_4c", InceptionModule(512, [160, 112, 224, 24, 64, 64], name + "Mixed_4c")),
            ("Mixed_4d", InceptionModule(512, [128, 128, 256, 24, 64, 64], name + "Mixed_4d")),
            ("Mixed_4e", InceptionModule(512, [112, 144, 288, 32, 64, 64], name + "Mixed_4e")),
            ("Mixed_4f", InceptionModule(528, [256, 160, 320, 32, 128, 128], name + "Mixed_4f")),
            (
                "MaxPool3d_5a_2x2",
                MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0),
            ),
            ("Mixed_5b", InceptionModule(832, [256, 160, 320, 32, 128, 128], name + "Mixed_5b")),
            ("Mixed_5c", InceptionModule(832, [384, 192, 384, 48, 128, 128], name + "Mixed_5c")),
        )

        for endpoint_name, endpoint in endpoints:
            self.end_points[endpoint_name] = endpoint
            if final_endpoint == endpoint_name:
                self.build()
                return

        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            1024,
            num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )
        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(
            1024,
            num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

    def build(self):
        for name, endpoint in self.end_points.items():
            self.add_module(name, endpoint)

    def forward(self, inputs):
        outputs = inputs
        for endpoint in self.VALID_ENDPOINTS:
            if endpoint in self.end_points:
                outputs = self._modules[endpoint](outputs)
        outputs = self.logits(self.dropout(self.avg_pool(outputs)))
        if self._spatial_squeeze:
            outputs = outputs.squeeze(3).squeeze(3)
        return outputs.mean(dim=2)

    def extract_features(self, inputs):
        outputs = inputs
        for endpoint in self.VALID_ENDPOINTS:
            if endpoint in self.end_points:
                outputs = self._modules[endpoint](outputs)
        return self.avg_pool(outputs)
