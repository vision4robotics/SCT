import torch.nn as nn
import torch
import time
class AlexNet(nn.Module):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), AlexNet.configs))
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
            )
        self.feature_size = configs[5]
        for param in self.layer1.parameters():
                param.requires_grad = False
        for param in self.layer2.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.layer1(x)
        x_0 = x
        x = self.layer2(x)
        x_1 = x
        x = self.layer3(x)
        x0 = x
        x1 = self.layer4(x)
        x = self.layer5(x1)
        return x_0, x_1, x0, x1, x

# x = torch.rand(1, 3, 256, 256).cuda()
# model = AlexNet().cuda()
# model.load_state_dict(torch.load('alexnet-bn.pth'))
# star = time.time()
# for i in range(100):
# # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
#     x_out = model(x)
# # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# # prof.export_chrome_trace('parallel.json')
# end = time.time() - star
# print(100/end)