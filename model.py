import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchvision.models as models
from torchsummary import summary


class DepthWiseConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(DepthWiseConv3D, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv3d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    groups=in_channels)
        # groups是一个数，当groups=in_channels时，表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv3d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out
#这个版本是针对两个模态的输入，输入通道为1
class Fusion_Block(nn.Module):
    def __init__(self):
        super(Fusion_Block, self).__init__()
    def forward(self, T1_weight , T2_weight):
        #(B, C, D, H, W)
        # 将T1_weight , T2_weight复制三份(16, 1, 13, 256, 256)->(16, 3, 13, 256, 256)
        T1_weight = T1_weight.repeat(1, 3, 1, 1, 1)
        T2_weight = T2_weight.repeat(1, 3 ,1, 1, 1)
        #print(T1_weight.shape , T2_weight.shape)

        #T1_weight , T2_weight按通道拼接成(16, 6, 13, 256, 256)
        fused_3d= torch.cat([T1_weight, T2_weight], dim=1)

        batch_size, channels, depth, height, width = fused_3d.shape
        # 将 (16, 6, 13, 256, 256) 转换为 (16*13, 6, 256, 256)
        fused_2d = fused_3d.permute(0, 2, 1, 3, 4).contiguous().view(-1, channels, height, width)#输入到resnet18的数据

        return fused_2d, fused_3d

#经过M_Module形状不会变化
class M_Module(nn.Module):
    def __init__(self, in_channels):
        super(M_Module, self).__init__()
        self.dconv = nn.Sequential(
            DepthWiseConv3D(in_channels, in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.gconv = nn.Sequential(
            nn.Conv3d(in_channels // 2, in_channels // 4, kernel_size=3, groups=in_channels // 4, padding=1),
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dconv1 = nn.Sequential(
            DepthWiseConv3D(in_channels//4, in_channels//4),
            nn.BatchNorm3d(in_channels//4),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout3d(p=0.1)
    #y1,y2,y3,p,q,s,t在网络图中均有描述
    def forward(self, x):
        y1 = self.dconv(x)
        y2 = torch.cat([x, y1], dim=1)
        p0, p1, p2, p3 = torch.chunk(y2, 4, dim=1)

        q1 = p1 + p0
        s1 = self.gconv(q1)
        t1 = self.dconv1(s1)
        v1 = torch.cat([t1, s1], dim=1)

        q2 = p2 + v1
        s2 = self.gconv(q2)
        t2 = self.dconv1(s2)
        v2 = torch.cat([t2, s2], dim=1)

        q3 = p3 + v2
        s3 = self.gconv(q3)
        t3 = self.dconv1(s3)
        v3 = torch.cat([t3, s3], dim=1)

        y3 = torch.cat([p0, v1, v2, v3], dim=1)
        y4 = self.conv(y3)
        y3 = self.dropout(y3)
        output = self.relu(y4 + x)
        return output

#找到距离value1最近的value2的因子
def find_nearest_divisible_factor(value1, value2):
        factors = []
        for_times = int(math.sqrt(value2))
        for i in range(for_times + 1)[1:]:
            if value2 % i == 0:
                factors.append(i)
                t = int(value2 / i)
                if not t == i:
                    factors.append(t)
        factors.sort()

        array = np.asarray(np.array(factors))
        idx = (np.abs(array - value1)).argmin()

        return array[idx]

#深度可分离卷积,用在BG Block的最后
class CT_Module(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CT_Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = 1
        if in_channels < out_channels:
            mid_channels = find_nearest_divisible_factor(in_channels, out_channels)
            # print(mid_channels)
            self.expand_conv = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.cheap_conv = nn.Sequential(
                DepthWiseConv3D(mid_channels, out_channels - mid_channels),
                nn.BatchNorm3d(out_channels - mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.compress_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        if self.in_channels < self.out_channels:
            out = self.expand_conv(x)
            out = torch.cat([out, self.cheap_conv(out)], 1)
        else:
            out = self.compress_conv(x)
        return out
class Down_Layer(nn.Module):

    def __init__(self, in_channels, out_channels,dilation = 1):
        super(Down_Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = 3
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size,stride = (1,2,2), dilation = (dilation, 1, 1), padding = (dilation,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
            out = self.down_conv(x)
            return out

#Stage = 3D Block = 3×3×3卷积 + 4 * M Module，Down_layer没有用
class Stage(nn.Module):

    def __init__(self,in_channels, out_channels, dilation = 1, num_block = 4):
        super(Stage,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = 3
        self.down_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride = (1, 2, 2), padding = (dilation, 1, 1), dilation = (dilation, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.m_layer = nn.Sequential(*[
            M_Module(out_channels)
            for _ in range(num_block)])
        self.dropout = nn.Dropout3d(p=0.1)
    def forward(self, x):
        x = self.down_layer(x)
        out = self.m_layer(x)
        out = self.dropout(out)

        return out

#经过BT_Blcok的形状不会发生变化
class BT_Block(nn.Module):
    def __init__(self,
                 slave_out_channels,
                 master_out_channels,
                 bt_mode = 'dual_mul',
                 drop_prob = 0.1):
        super(BT_Block,self).__init__()
        assert bt_mode in ["dual_mul", "dual_add",
                           "s2m_mul", "s2m_add",
                           "m2s_mul", "m2s_add",
                           "dual_mul_add", "dual_add_mul",
                           None,
                           "s2m_mul_e", "s2m_add_e",
                           "m2s_mul_e", "m2s_add_e",
                           "dual_mul_e", "dual_add_e",
                           "dual_mul_add_e", "dual_add_mul_e"]
        self.bt_mode = bt_mode
        #Resize Module
        self.resize_convs = nn.Identity()
        #Mix Module
        self.mix_conv = nn.Sequential(
            nn.Conv3d(slave_out_channels, slave_out_channels, kernel_size=(3,1,1),stride = (1,1,1),padding =(1,0,0)),
            nn.BatchNorm3d(slave_out_channels),
            nn.ReLU(inplace=True)
        )
        #CT Module
        if bt_mode == "dual_mul" or bt_mode == "dual_add":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                )
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                )
        elif bt_mode == "dual_add_mul":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                )
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                )
        elif bt_mode == "dual_mul_add":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                )
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                )
        elif bt_mode == "s2m_mul" or bt_mode == "s2m_add":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                )
        elif bt_mode == "m2s_mul" or bt_mode == "m2s_add":
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                )
        elif bt_mode == "dual_mul_e" or bt_mode == "dual_add_e":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                )
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                )
            self.unify_conv = \
                DepthWiseConv3D(
                    slave_out_channels,
                    master_out_channels,
                    kernel_size = 1,
                    padding = 0,
                )
            self.gather_conv = \
                DepthWiseConv3D(
                    master_out_channels,
                    master_out_channels,
                )
        elif bt_mode == "dual_add_mul_e":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                )
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                )
            self.unify_conv = \
                DepthWiseConv3D(
                    slave_out_channels,
                    master_out_channels,
                    kernel_size = 1,
                    padding = 0,
                )
            self.gather_conv = \
                DepthWiseConv3D(
                    master_out_channels,
                    master_out_channels,
                )
        elif bt_mode == "dual_mul_add_e":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                )
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                )
            self.unify_conv = \
                DepthWiseConv3D(
                    slave_out_channels,
                    master_out_channels,
                    kernel_size = 1,
                    padding = 0,
                )
            self.gather_conv = \
                DepthWiseConv3D(
                    master_out_channels,
                    master_out_channels,
                )
        elif bt_mode == "s2m_mul_e" or bt_mode == "s2m_add_e":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                )
            self.unify_conv = \
                DepthWiseConv3D(
                    slave_out_channels,
                    master_out_channels,
                    kernel_size=1,
                    padding=0,
                )
            self.gather_conv = \
                DepthWiseConv3D(
                    master_out_channels,
                    master_out_channels,
                )
        elif bt_mode == "m2s_mul_e" or bt_mode == "m2s_add_e":
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                )
            self.unify_conv = \
                DepthWiseConv3D(
                    slave_out_channels,
                    master_out_channels,
                    kernel_size=1,
                    padding=0,
                )
            self.gather_conv = \
                DepthWiseConv3D(
                    master_out_channels,
                    master_out_channels,
                )
        else:
            self.s2m_conv = nn.Identity()
            self.m2s_conv = nn.Identity()
        self.drop_path = nn.Dropout3d(drop_prob) if drop_prob > 0.0 else nn.Identity()
    def forward(self,x):
        x_slave, x_master = x
        _, c1, h1, w1 = x_slave.shape
        bs2, c2, d2, h2, w2 = x_master.shape
        #改这里的深度
        x_slave = x_slave.view(bs2, 152, c1, h1, w1).permute(0, 2, 1, 3, 4)

        # x_slave_list, x_master = x[0], x[1]
        # # depth的concat->C
        # bs, channel, height, width = x_slave_list[0].shape
        # #x_slave_list是一个有13个数据的列表,列表中数据形状为(16,6,256,256)
        # #x_master是一个3D的输入形状为(16, 6, 13, 256, 256)
        # #要先把x_slave_list列表循环把数据拿出,然后再拼接成形状为(16, 6, 13, 256, 256)的数据
        # x_slave_reshaped = [arr.reshape(bs, channel, 1, height, width) for arr in x_slave_list]#(16, 6, 1, 256, 256)
        # x_slave = torch.cat(x_slave_reshaped, dim=2)
        #
        # # 提取原始形状信息
        # batch_size, channels0, original_depth0, height0, width0= x_slave.shape
        # batch_size, channels1, original_depth1, height1, width1 = x_master.shape

        #Resize Module三线性插值
        x_slave2m = torch.nn.functional.interpolate(x_slave, size=(d2, h2, w2), mode='trilinear', align_corners=False)
        x_master2s = torch.nn.functional.interpolate(x_master, size=(152, h1, w1), mode='trilinear', align_corners=False)

        # 从这之后x_slave,x_master,都是3D的了
        if self.bt_mode == "dual_mul" or self.bt_mode == "dual_add":
            out_master = self.s2m_conv(self.mix_conv(x_slave2m))
            out_master = self.drop_path(out_master)
            if "mul" in self.bt_mode:
                out_master = x_master * out_master
            else:
                out_master = x_master + out_master

            out_slave = self.m2s_conv(x_master2s)
            out_slave = self.drop_path(out_slave)
            if "mul" in self.bt_mode:
                out_slave = x_slave * out_slave
            else:
                out_slave = x_slave + out_slave

            out_slave = out_slave.permute(0, 2, 1, 3, 4).contiguous().view(-1, c1, h1, w1)
            results = ((out_slave, out_master))
        elif self.bt_mode == "dual_add_mul":
            out_master = self.s2m_conv(self.mix_conv(x_slave2m))
            out_master = self.drop_path(out_master)
            out_master = x_master + out_master

            out_slave = self.m2s_conv(x_master2s)
            out_slave = self.drop_path(out_slave)
            out_slave = x_slave * out_slave
            out_slave = out_slave.permute(0, 2, 1, 3, 4).contiguous().view(-1, c1, h1, w1)
            results = ((out_slave, out_master))
        elif self.bt_mode == "s2m_mul" or self.bt_mode == "s2m_add":
            out_master = self.s2m_conv(self.mix_conv(x_slave2m))
            if "mul" in self.bt_mode:
                out_master = x_master * out_master
            else:
                out_master = x_master + out_master
            out_master = self.drop_path(out_master)


            out_slave = self.drop_path(x_slave)
            out_slave = out_slave.permute(0, 2, 1, 3, 4).contiguous().view(-1, c1, h1, w1)
            results = ((out_slave, out_master))
        elif self.bt_mode == "m2s_mul" or self.bt_mode == "m2s_add":
            out_slave = self.m2s_conv(x_master2s)

            if "mul" in self.bt_mode:
                out_slave = x_slave * out_slave
            else:
                out_slave = x_slave + out_slave
            out_slave = self.drop_path(out_slave)
            out_slave = out_slave.permute(0, 2, 1, 3, 4).contiguous().view(-1, c1, h1, w1) 
            out_master = self.drop_path(x_master)
            results = ((out_slave, out_master))

        elif self.bt_mode == "dual_mul_e" or self.bt_mode == "dual_add_e":
            out_master = self.s2m_conv(self.mix_conv(x_slave2m))
            out_master = self.drop_path(out_master)
            if "mul" in self.bt_mode:
                out_master = x_master * out_master
            else:
                out_master = x_master + out_master
            out_slave = self.m2s_conv(x_master2s)
            out_slave = self.drop_path(out_slave)
            if "mul" in self.bt_mode:
                out_slave = x_slave * out_slave
            else:
                out_slave = x_slave + out_slave

            out_slave = torch.nn.functional.interpolate(out_slave, size=(d2, h2, w2), mode='trilinear', align_corners=False)
            out_slave = self.s2m_conv(self.mix_conv(out_slave))
            out_slave = self.drop_path(out_slave)
            results = self.gather_conv(out_master + out_slave)
            results = (self.drop_path(results))
        elif self.bt_mode == "dual_add_mul_e":
            out_master = self.s2m_conv(self.mix_conv(x_slave2m))
            out_master = self.drop_path(out_master)
            out_master = x_master + out_master

            out_slave = self.m2s_conv(x_master2s)
            out_slave = self.drop_path(out_slave)
            out_slave = x_slave * out_slave

            out_slave = torch.nn.functional.interpolate(out_slave, size=(d2, h2, w2), mode='trilinear', align_corners=False)
            out_slave = self.s2m_conv(self.mix_conv(out_slave))
            out_slave = self.drop_path(out_slave)
            results = self.gather_conv(out_master + out_slave)

            results = (self.drop_path(results))
        elif self.bt_mode == "s2m_mul_e" or self.bt_mode == "s2m_add_e":
            out_master = self.s2m_conv(self.mix_conv(x_slave2m))
            out_master = self.drop_path(out_master)
            if "mul" in self.bt_mode:
                out_master = x_master * out_master
            else:
                out_master = x_master + out_master

            out_slave = torch.nn.functional.interpolate(out_slave, size=(d2, h2, w2), mode='trilinear', align_corners=False)
            out_slave = self.s2m_conv(self.mix_conv(out_slave))
            out_slave = self.drop_path(out_slave)
            results = self.gather_conv(out_master + out_slave)

            results = (self.drop_path(results))

        elif self.bt_mode == "m2s_mul_e" or self.bt_mode == "m2s_add_e":
            out_slave = self.m2s_conv(x_master2s)
            out_slave = self.drop_path(out_slave)

            if "mul" in self.bt_mode:
                out_slave = x_slave * out_slave
            else:
                out_slave = x_slave + out_slave

            out_slave = torch.nn.functional.interpolate(out_slave, size=(d2, h2, w2), mode='trilinear', align_corners=False)
            out_slave = self.s2m_conv(self.mix_conv(out_slave))
            out_slave = self.drop_path(out_slave)
            results = self.gather_conv(x_master + out_slave)
            
            results = (self.drop_path(results))
        else:
            results = (self.drop_path(self.s2m_conv(x_slave)),
                self.drop_path(self.m2s_conv(x_master)))

        return results
class Twist_ResNet_3D(nn.Module):
    def __init__(self,
                 branch1_channels = (64, 128, 256, 512),
                 branch2_channels=(32, 64, 128, 256),
                 branch2_dilations=(1, 2, 4, 8),
                 bt_modes= ('dual_mul', 'dual_mul', 'dual_mul', 'dual_mul_e'),
                 multi_modals=2,  #默认数据集HPC数据集，双模态
                 num_classes = 2,
                 stem_channels=64):
        super(Twist_ResNet_3D, self).__init__()
        self.multi_modals = multi_modals
        self.stem_channels = stem_channels
        self.num_classes = num_classes
        self.bt_stage = []
        self.fusion_block = Fusion_Block()
        self.dropout = nn.Dropout3d(p=0.1)
        # #分支1 - 使用ResNet18预训练模型
        # self.branch1 = models.resnet18(pretrained=True)
        # #输入两个模态，所以resnet输入通道需要×2
        # self.branch1.conv1 = nn.Conv2d(multi_modals * 3, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)# 修改输入通道数为6
        # #去掉池化层
        # self.branch1.avgpool = nn.Identity()
        # #去掉resnet18的全连接层
        # self.branch1.fc = nn.Identity()

        # branch1: 使用预训练的 ResNet-18
        resnet18 = models.resnet18(pretrained=True)
        self.branch1_stem = nn.Sequential(
            nn.Conv2d(multi_modals * 3, 64, kernel_size=7, stride=2, padding=3),
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool
        )
        self.branch1_layer1 = resnet18.layer1
        self.branch1_layer2 = resnet18.layer2
        self.branch1_layer3 = resnet18.layer3
        self.branch1_layer4 = resnet18.layer4
        self.branch1 = nn.Sequential(
            self.branch1_stem,
            self.branch1_layer1,
            self.branch1_layer2,
            self.branch1_layer3,
            self.branch1_layer4,
        )

        self.branch2_stem = nn.Sequential(
            nn.Conv3d(multi_modals * 3, stem_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(stem_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.branch2_layer1 = Stage(
            in_channels=branch2_channels[0] ,
            out_channels=branch2_channels[0],
            dilation=branch2_dilations[0],
        )
        self.branch2_layer2 = Stage(
            in_channels=branch2_channels[0],
            out_channels=branch2_channels[1],
            dilation=branch2_dilations[1],
        )
        self.branch2_layer3 = Stage(
            in_channels=branch2_channels[1],
            out_channels=branch2_channels[2],
            dilation=branch2_dilations[2],
        )
        self.branch2_layer4 = Stage(
            in_channels=branch2_channels[2],
            out_channels=branch2_channels[3],
            dilation=branch2_dilations[3],
        )
        #分支2 - 3D分支
        self.branch2 = nn.Sequential(
            self.branch2_stem,
            self.branch2_layer1,
            self.branch2_layer2,
            self.branch2_layer3,
            self.branch2_layer4,
        )

        #BT
        self.bt_module_1 = BT_Block(
            slave_out_channels=branch1_channels[0],
            master_out_channels=branch2_channels[0],
            bt_mode=bt_modes[0],
        )
        self.bt_module_2 = BT_Block(
            slave_out_channels=branch1_channels[1],
            master_out_channels=branch2_channels[1],
            bt_mode=bt_modes[1],
        )
        self.bt_module_3 = BT_Block(
            slave_out_channels=branch1_channels[2],
            master_out_channels=branch2_channels[2],
            bt_mode=bt_modes[2],
        )
        self.bt_stage = nn.Sequential(
            self.bt_module_1,
            self.bt_module_2,
            self.bt_module_3,
        )

        #BG
        self.bg_module = BT_Block(
            slave_out_channels=branch1_channels[3],
            master_out_channels=branch2_channels[3],
            bt_mode=bt_modes[3],
        )
        #3D Classification Head

        #3D GAP
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  # 3D Global Average Pooling
        #3D FC
        self.fc = nn.Linear(256, 2)  # Fully Connected layer

        # #Softmax
        # self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def _initialize_weights(self):
        # # 初始化 branch1 的第一个卷积层权重
        # resnet18_pretrained = models.resnet18(pretrained=True)
        # self.branch1.load_state_dict(resnet18_pretrained.state_dict(), strict=False)
        # conv1_weight = resnet18_pretrained.conv1.weight.data
        # new_conv1_weight = torch.cat([conv1_weight] * 2, dim=1) / 2  # 将权重重复一次，并进行平均
        # self.branch1.conv1.weight.data = new_conv1_weight

        # # 初始化 branch1 的第一个卷积层权重
        # resnet18_pretrained = models.resnet18(pretrained=True)
        # self.branch1.load_state_dict(resnet18_pretrained.state_dict(), strict=False)
        # conv1_weight = resnet18_pretrained.conv1.weight.data
        # new_conv1_weight = torch.cat([conv1_weight] * 2, dim=1) / 2  # 将权重重复一次，并进行平均
        # self.branch1[0].weight.data = new_conv1_weight

        resnet18_pretrained = models.resnet18(pretrained=True)
        # 初始化 branch1 的第一个卷积层权重
        conv1_weight = resnet18_pretrained.conv1.weight.data
        new_conv1_weight = torch.cat([conv1_weight] * 2, dim=1) / 2  # 将权重重复一次，并进行平均
        self.branch1_stem[0].weight.data = new_conv1_weight

        # 初始化 BatchNorm 层的权重和偏置
        self.branch1_stem[1].weight.data = resnet18_pretrained.bn1.weight.data
        self.branch1_stem[1].bias.data = resnet18_pretrained.bn1.bias.data

        # 初始化 branch2 的权重
        for m in self.branch2.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 初始化 processing_module 和 fc 层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # print("After Input:", type(x1), x1[0].shape, type(x2), x2.shape)
        x1 = x1.permute(0, 2, 1, 3)
        x2 = x2.permute(0, 2, 1, 3)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        # print("After Input:", type(x1), x1[0].shape, type(x2), x2.shape)

        #Fusion Block
        out1, out2 = self.fusion_block(x1, x2)
        # print("After Fusion Block:", type(out1), out1[0].shape, type(out2), out2.shape)

        # Stem
        out1 = self.branch1_stem(out1)
        out2 = self.branch2_stem(out2)
        # print("After Stem:", type(out1), out1.shape, type(out2), out2.shape)

        # Layer 1
        out1 = self.branch1_layer1(out1)
        out2 = self.branch2_layer1(out2)
        out1, out2 = self.bt_module_1((out1, out2))
        # print("After Layer 1:", type(out1), out1.shape, type(out2), out2.shape)

        # Layer 2
        out1 = self.branch1_layer2(out1)
        out2 = self.branch2_layer2(out2)
        out1, out2 = self.bt_module_2((out1, out2))
        # print("After Layer 2:", type(out1), out1.shape, type(out2), out2.shape)

        # Layer 3
        out1 = self.branch1_layer3(out1)
        out2 = self.branch2_layer3(out2)
        out1, out2 = self.bt_module_3((out1, out2))
        # print("After Layer 3:", type(out1), out1.shape, type(out2), out2.shape)

        # Layer 4
        out1 = self.branch1_layer4(out1)
        out2 = self.branch2_layer4(out2)
        # print("After Layer 4:", type(out1), out1.shape, type(out2), out2.shape)
        # BG
        out = self.bg_module((out1, out2))
        # print("After bg:", type(out), out.shape)
        out = self.dropout(out)
        # 3D GAP
        out = self.gap(out)

        # Flatten, shape (batch_size, 256)
        out = out.view(out.size(0), -1) 
        # 3D FC
        out = self.fc(out)

        # Softmax
        out = F.softmax(out, dim=1)

        return out



def main():
       

   # 初始化模型
    model = Twist_ResNet_3D()

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型移到 GPU 或 CPU
    model = model.to(device)

    # 示例输入
    input1 = torch.randn( 2, 152, 244, 244).to(device)
    input2 = torch.randn( 2, 152, 244, 244).to(device)

    # 前向传播
    out = model(input1, input2)
    print('==============================================================')
    print('out', out.shape)

    # 打印网络结构
    # summary(model, input_size=[(1, 13, 256, 256), (1, 13, 256, 256)], device=device.type)



if __name__ == "__main__":
    main()