import torch.nn as nn
import torch.nn.functional as F
import torch

# based on the paper: 'U-Net convolutional networks for biomedical image segmentation'
class ConvRelu(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, filter_size, padding):
        # in the original paper this padding=0
        super(ConvRelu, self).__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(n_in_channels, n_out_channels, filter_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, conv_relu_input):
        output = self.operation(conv_relu_input)
        return output

class ConvReluSeq(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_components=2, filter_size=3, padding=1):
        # in the original paper this padding=0
        super(ConvReluSeq, self).__init__()
        components = [ConvRelu(n_in_channels, n_out_channels, filter_size, padding)]
        for i in xrange(1,n_components):
            components.append(ConvRelu(n_out_channels, n_out_channels, filter_size, padding))
        self.operation = nn.Sequential(*components)

    def forward(self, conv_relu_input):
        output = self.operation(conv_relu_input)
        return output

class UpConvWithCopyCrop(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, filter_size=2, stride=2):
        super(UpConvWithCopyCrop, self).__init__()
        self.up = nn.ConvTranspose2d(n_in_channels, n_out_channels, filter_size, stride=stride)

    def forward(self, up_conv_input, copy_crop_input):
        up_conv_output = self.up(up_conv_input)
        # tensor is C X H x W
        height_diff = up_conv_output.size()[2] - copy_crop_input.size()[2]
        width_diff = up_conv_output.size()[3] - copy_crop_input.size()[3]
        copy_crop_output = F.pad(copy_crop_input, (height_diff // 2, int(height_diff / 2),
                                 width_diff // 2, int(width_diff/ 2)))
        # concatenate them together
        output = torch.cat([copy_crop_output, up_conv_output], dim=1)
        return output

class UNet(nn.Module):
    def __init__(self, n_in_channels, n_classes):
        #operations for the contracting path
        super(UNet, self).__init__()
        self.conv_relu_seq_contract_1 = ConvReluSeq(n_in_channels, 64, filter_size=3, n_components=2) # a sequence of 3X3 conv                                                                                         # followed by Relu (default params)
        self.conv_relu_seq_contract_2 = ConvReluSeq(64, 128)
        self.conv_relu_seq_contract_3 = ConvReluSeq(128, 256)
        self.conv_relu_seq_contract_4 = ConvReluSeq(256, 512)
        self.conv_relu_seq_contract_5 = ConvReluSeq(512, 1024)
        self.max_pool = nn.MaxPool2d(2)  # stride = 2

        # operations for the expansive path
        self.up_conv_with_copy_crop_1 = UpConvWithCopyCrop(1024, 512, filter_size=2, stride=2)
        self.conv_relu_seq_expan_1= ConvReluSeq(1024, 512)
        self.up_conv_with_copy_crop_2 = UpConvWithCopyCrop(512, 256)
        self.conv_relu_seq_expan_2 = ConvReluSeq(512, 256)
        self.up_conv_with_copy_crop_3 = UpConvWithCopyCrop(256, 128)
        self.conv_relu_seq_expan_3 = ConvReluSeq(256, 128)
        self.up_conv_with_copy_crop_4 = UpConvWithCopyCrop(128, 64)
        self.conv_relu_seq_expan_4 = ConvReluSeq(128, 64)
        self.out_conv = nn.Conv2d(64, n_classes, 3)


    def forward(self, input):
        # contracting path
        contract_out_1 = self.conv_relu_seq_contract_1(input)
        contract_out_2 = self.conv_relu_seq_contract_2(self.max_pool(contract_out_1))
        contract_out_3 = self.conv_relu_seq_contract_3(self.max_pool(contract_out_2))
        contract_out_4 = self.conv_relu_seq_contract_4(self.max_pool(contract_out_3))
        contract_out_5 = self.conv_relu_seq_contract_5(self.max_pool(contract_out_4))
        # expansive path
        expansive_out = self.conv_relu_seq_expan_1(self.up_conv_with_copy_crop_1(contract_out_5, contract_out_4))
        expansive_out = self.conv_relu_seq_expan_2(self.up_conv_with_copy_crop_2(expansive_out, contract_out_3))
        expansive_out = self.conv_relu_seq_expan_3(self.up_conv_with_copy_crop_3(expansive_out, contract_out_2))
        expansive_out = self.conv_relu_seq_expan_4(self.up_conv_with_copy_crop_4(expansive_out, contract_out_1))
        expansive_out = self.out_conv(expansive_out)
        # transform to range [0,1]
        output = torch.nn.functional.sigmoid(expansive_out)
        return output




