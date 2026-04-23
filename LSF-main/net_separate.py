import torch.nn as nn
import torch
from function import calc_mean_std

class decoder(nn.Module):
    def __init__(self, chan=1):
        super(decoder, self).__init__()
        self.up1 = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                    nn.Conv2d(512, 256, (3, 3)),
                                    nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                    nn.Conv2d(256, 256, (3, 3)),
                                    nn.ReLU(),
                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                    nn.Conv2d(256, 256, (3, 3)),
                                    nn.ReLU(),
                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                    nn.Conv2d(256, 256, (3, 3)),
                                    nn.ReLU())
        self.up2 = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                    nn.Conv2d(256, 128, (3, 3)),
                                    nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                    nn.Conv2d(128, 128, (3, 3)),
                                    nn.ReLU())
        self.up3 = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                    nn.Conv2d(128, 64, (3, 3)),
                                    nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                    nn.Conv2d(64, 64, (3, 3)),
                                    nn.ReLU(),
                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                    nn.Conv2d(64, 3, (3, 3)))

    def forward(self, fea):
        out = self.up1(fea)
        out = self.up2(out)
        cs = self.up3(out)
        return cs

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

def initialize_weights(net):
    for m in net.modules():
        try:
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 0.02)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                # m.weight.data.normal_(0, 0.02)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.02)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        except Exception as e:
            # print(f'SKip layer {m}, {e}')
            pass

class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
            kernel_size=1, stride=1, bias=bias)

        self.ins_norm1 = nn.InstanceNorm2d(in_channels)
        self.activation1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.ins_norm1(out)
        out = self.activation1(out)
        out = self.pointwise(out)
        return out

class remove(nn.Module):
    def __init__(self, chan=512):
        super(remove, self).__init__()
        self.xxx = nn.Sequential(SeparableConv2D(chan, chan),
                                 nn.InstanceNorm2d(chan),
                                 nn.ReLU(inplace=True),

                                 SeparableConv2D(chan, chan),
                                 nn.InstanceNorm2d(chan),
                                 nn.ReLU(inplace=True),

                                 SeparableConv2D(chan, chan),)

    def forward(self, feature):
        content = self.xxx(feature)
        style = feature - content
        return content, style

# all
class adain(nn.Module):
    def __init__(self, channl=512):
        super(adain, self).__init__()
        self.front_patch = 16
        self.style_encoder = nn.Sequential(Conv2dBlock(channl, channl, 4, 2, 1, norm='none', activation='relu', pad_type='reflect'),
                                           nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(channl, channl, (1, 1)))
        self.weight = nn.Conv2d(channl, channl, (1, 1))
        self.bias = nn.Conv2d(channl, channl, (1, 1))
        self.conv_fusion = nn.Conv2d(channl*2, channl, (1, 1))
        self.conv_combine = nn.Conv2d(self.front_patch, 1, (1, 1))

    def forward(self, content, style):

        pre_style = self.style_encoder(style)
        style_weight = self.weight(pre_style)
        style_bias = self.bias(pre_style)
        norm_content = mean_variance_norm(content)
        content_part1 = norm_content * style_weight + style_bias

        B, C, H, W = content.size()
        norm_content2 = mean_variance_norm(content)
        norm_style2 = mean_variance_norm(style)
        attn = torch.bmm(norm_content2.view(B, C, H*W).permute(0, 2, 1), norm_style2.view(B, C, H*W))
        _, sort_index = torch.sort(attn, dim=2, descending=True)
        select_id = sort_index[:, :, :self.front_patch]

        deal_style = style.view(B, C, H*W)

        style_arrange = deal_style.unsqueeze(0).repeat(H*W, 1, 1, 1)[torch.arange(H*W)[:, None], torch.arange(B)[:, None, None], :, select_id].permute(0, 2, 3, 1)

        style_arrange = self.conv_combine(style_arrange).view(B, C, H, W)
        content_part2 = content + style_arrange

        output = self.conv_fusion(torch.cat([content_part1, content_part2], 1))
        return output

class transfm(nn.Module):
    def __init__(self, tong=512):
        super(transfm, self).__init__()
        self.remove_module = remove(512)
        self.fusion_module = adain(512)

    def forward(self, Content, Style):
        Content_content, Content_style = self.remove_module(Content)
        Style_content, Style_style = self.remove_module(Style)

        cs = self.fusion_module(Content_content, Style_style)

        cc = self.fusion_module(Content_content, Content_style)
        ss = self.fusion_module(Style_content, Style_style)
        return cs, cc, ss, Content_content, Content_style, Style_content, Style_style

class Net(nn.Module):
    def __init__(self, encoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 64, (1, 1)))

        self.decoder = decoder()
        self.transform = transfm()
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def feature_contrastive(self, input):
        out = self.proj(input)
        return out

    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        t, cc_t, ss_t, content_content, content_style, style_content, style_style = self.transform(content_feats[-1], style_feats[-1])
        l_fen = torch.sum(torch.abs(self.feature_contrastive(content_content) * self.feature_contrastive(content_style))) \
                + torch.sum(torch.abs(self.feature_contrastive(style_content) * self.feature_contrastive(style_style)))

        g_t = self.decoder(t)
        cc_g_t = self.decoder(cc_t)
        ss_g_t = self.decoder(ss_t)

        Fcc = self.encode_with_intermediate(cc_g_t)
        Fss = self.encode_with_intermediate(ss_g_t)
        l_identity1 = self.calc_content_loss(cc_g_t, content) + self.calc_content_loss(ss_g_t, style)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 4):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i], style_feats[i])

        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(mean_variance_norm(g_t_feats[-1]), mean_variance_norm(content_feats[-1]))
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s, l_identity1, l_identity2, l_fen, g_t
