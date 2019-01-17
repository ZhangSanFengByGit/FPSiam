from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamRPN(nn.Module):
    def __init__(self, size=1, feature_out=256, anchor=5, pretrain=False):
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x==3 else x*size, configs))
        feat_in = configs[-1]
        super(SiamRPN, self).__init__()
        self.featureExtract = nn.Sequential(  #271 127
            nn.Conv2d(configs[0], configs[1] , kernel_size=11, stride=2),  #131 59
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),  #65   29
            nn.ReLU(inplace=True),

            nn.Conv2d(configs[1], configs[2], kernel_size=5), #61   25
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),  #30   12
            nn.ReLU(inplace=True),

            nn.Conv2d(configs[2], configs[3], kernel_size=3), #28 10
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),

            nn.Conv2d(configs[3], configs[4], kernel_size=3), #26  8   11
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3), #24  6
            nn.BatchNorm2d(configs[5])
        )

        self.anchor = anchor
        self.feature_out = feature_out

        self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3) #4
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3) #22
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3) #4
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3) #22
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)

        self.r1_kernel = []
        self.cls1_kernel = []

        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'instance_size': 271, 'adaptive': False} # 0.655

        self.old_f = []
        self.temple_norm = []

        self.conv_xfit = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv_oldfit = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.kernel_pre = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 25, kernel_size=1, stride=1)
        )

        self.embed_net = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 2048, kernel_size=1, stride=1),
            nn.BatchNorm2d(2048)
        )

        self.delta = [[-2,-2], [-1,-2], [0,-2], [1,-2], [2,-2], \
                 [-2,-1], [-1,-1], [0,-1], [1,-1], [2,-1], \
                 [-2,0], [-1,0], [0,0], [1,0], [2,0], \
                 [-2,1], [-1,1], [0,1], [1,1], [2,1], \
                 [-2,2], [-1,2], [0,2], [1,2], [2,2]]

        if pretrain==False:
            self.init()


    def forward(self, x, set_source=False):
        assert not torch.isnan(x).any()

        batch_size = x.size(0)
        x_f = self.featureExtract(x)

        if set_source:
            self.old_f = x_f.detach()
            return

        b,c,h,w = x_f.size(0),x_f.size(1),x_f.size(2),x_f.size(3)

        if type(self.old_f)==torch.Tensor:
            x_f_fit = self.conv_xfit(x_f)
            old_fit = self.conv_oldfit(self.old_f)
            concat_f = torch.cat((x_f_fit, old_fit), dim=1)
            kernels = self.kernel_pre(concat_f)
            assert not torch.isnan(kernels).any()

            kernels = kernels.permute(0,2,3,1).contiguous().view(b, h*w, 5*5)
            kernels = F.softmax(kernels, dim=2).view(b, h*w, 5, 5)
            prop_f = self.propagate(kernels,b,c,h,w)
            assert not torch.isnan(prop_f).any()

            embed_prop = self.embed_net(prop_f)
            embed_x = self.embed_net(x_f)

            prop_w = self.compute_weigt(embed_prop)
            assert not torch.isnan(prop_w).any()

            x_w = self.compute_weigt(embed_x)
            assert not torch.isnan(x_w).any()

            weights = F.softmax(torch.cat([prop_w.unsqueeze(0), x_w.unsqueeze(0)], dim=0), dim=0)
                      #[torch.log(prop_w)/(torch.log(prop_w)+torch.log(x_w)) , torch.log(x_w)/(torch.log(prop_w)+torch.log(x_w))]
            assert not torch.isnan(weights[0]).any()
            assert not torch.isnan(weights[1]).any()

            prop_f_weighted = torch.empty(prop_f.size(), device='cuda', requires_grad = False)
            x_f_weighted = torch.empty(x_f.size(), device='cuda', requires_grad = False)
            for batch in range(batch_size):
                prop_f_weighted[batch] = prop_f[batch] * weights[0,batch]
                x_f_weighted[batch] = x_f[batch] * weights[1,batch]

            fuzed_x_f = prop_f_weighted + x_f_weighted

            print('weights_0: {}'.format(weights[0]))
            print('weights_1: {}'.format(weights[1]))
            self.old_f = fuzed_x_f.detach()
            assert not torch.isnan(fuzed_x_f).any()

        else:
            fuzed_x_f = x_f
            self.old_f = fuzed_x_f.detach()
            assert not torch.isnan(fuzed_x_f).any()

        delta = torch.empty([batch_size, 4*5, 19, 19], device='cuda', requires_grad = False)
        score = torch.empty([batch_size, 2*5, 19, 19], device='cuda', requires_grad = False)

        for batch in range(batch_size):
            delta[batch] = self.regress_adjust(F.conv2d(self.conv_r2(fuzed_x_f[batch].unsqueeze(0)), self.r1_kernel[batch]))
            score[batch] = F.conv2d(self.conv_cls2(fuzed_x_f[batch].unsqueeze(0)), self.cls1_kernel[batch])

        assert not torch.isnan(delta).any()
        assert not torch.isnan(score).any()
        return delta,score
        #self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), \
               #F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)


    def temple(self, z, z_large):
        assert not torch.isnan(z).any()
        assert not torch.isnan(z_large).any()
        self.old_f = None
        batch_size = z.size(0)

        z_f_large = self.featureExtract(z_large)
        assert not torch.isnan(z_f_large).any()
        embed_z = self.embed_net(z_f_large)
        self.temple_norm = F.normalize(embed_z, p=2, dim=1)

        z_f = self.featureExtract(z)
        assert not torch.isnan(z_f).any()
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(batch_size, self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(batch_size, self.anchor*2, self.feature_out, kernel_size, kernel_size)
        assert self.r1_kernel.size() == torch.Size([batch_size, 20, 256, 4, 4]), '{}'.format(self.r1_kernel.size())


    def propagate(self, kernels,b,c,h,w):
        prop_f = torch.empty([b,c,h,w], device='cuda', requires_grad = False)
        for bb in range(b):  
            for hh in range(h):   #24
                for ww in range(w):   #24
                    product = None
                    for order in range(5*5):   #25
                        if hh+self.delta[order][1]>=0 and hh+self.delta[order][1]<h \
                        and ww+self.delta[order][0]>=0 and ww+self.delta[order][0]<w:
                            cur = self.old_f[bb,:,hh+self.delta[order][1], ww+self.delta[order][0]] * \
                                  kernels[ bb, hh*w+ww, int(order/5), int(order%5) ]

                            if type(product)!=torch.Tensor:
                                product = cur
                            else:
                                product = product+cur

                    prop_f[bb,:, hh, ww] = product

        #prop_f.register_hook(print)
        return prop_f


    def compute_weigt(self, embed_f):
        batch_size = embed_f.size(0)
        embed_norm = F.normalize(embed_f, p=2, dim=1)
        product = embed_norm * self.temple_norm
        weight = torch.mean(torch.sum(product, dim=1).view(batch_size, product.size(-1)* product.size(-1)), dim=1)
        assert weight.size()==torch.Size([batch_size]),'weight.size {}'.format(weight.size())
        return weight


    def init(self):
        for instance in self.conv_xfit.modules():
            if isinstance(instance, nn.Conv2d):
                nn.init.kaiming_normal_(instance.weight, mode='fan_out')
                if instance.bias is not None:
                    nn.init.constant_(instance.bias, 0)
            elif isinstance(instance, nn.BatchNorm2d):
                nn.init.constant_(instance.weight, 1)
                nn.init.constant_(instance.bias, 0)


        for instance in self.conv_oldfit.modules():
            if isinstance(instance, nn.Conv2d):
                nn.init.kaiming_normal_(instance.weight, mode='fan_out')
                if instance.bias is not None:
                    nn.init.constant_(instance.bias, 0)
            elif isinstance(instance, nn.BatchNorm2d):
                nn.init.constant_(instance.weight, 1)
                nn.init.constant_(instance.bias, 0)


        for instance in self.kernel_pre.modules():
            if isinstance(instance, nn.Conv2d):
                nn.init.kaiming_normal_(instance.weight, mode='fan_out')
                if instance.bias is not None:
                    nn.init.constant_(instance.bias, 0)
            elif isinstance(instance, nn.BatchNorm2d):
                nn.init.constant_(instance.weight, 1)
                nn.init.constant_(instance.bias, 0)


        for instance in self.embed_net.modules():
            if isinstance(instance, nn.Conv2d):
                nn.init.kaiming_normal_(instance.weight, mode='fan_out')
                if instance.bias is not None:
                    nn.init.constant_(instance.bias, 0)
            elif isinstance(instance, nn.BatchNorm2d):
                nn.init.constant_(instance.weight, 1)
                nn.init.constant_(instance.bias, 0)

'''
            elif isinstance(instance, nn.BatchNorm2d):
                nn.init.constant(instance.weight, 1)
                nn.init.constant(instance.bias, 0)

            elif isinstance(instance, nn.Linear):
                nn.init.normal(instance.weight, std=0.001)
                if instance.bias is not None:
                    nn.init.constant(instance.bias, 0)
'''

