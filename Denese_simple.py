import torch
import torch.nn as nn


class ada_channel_conv(nn.Module):
    def __init__(self):
        super(ada_channel_conv, self).__init__()

    def forward(self, x):
        conv = nn.Conv2d(in_channels=x.shape[1], out_channels=64, kernel_size=3, padding=1, stride=1, bias=True)
        x = conv(x)
        return x

class net1x1(nn.Module):
    def __init__(self):
        super(net1x1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=64,kernel_size=3,padding=1,stride=1,bias=True)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1,bias=True)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1,bias=True)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,padding=1,stride=1,bias=True)
        self.relu = nn.ReLU()
        
        #Concat
        self.add_conv_1 = nn.Conv2d(in_channels=66,out_channels=64,kernel_size=3,padding=1,stride=1,bias=True)

        self.add_conv_2 = nn.Conv2d(in_channels=130,out_channels=64,kernel_size=3,padding=1,stride=1,bias=True)
        self.add_conv_3 = nn.Conv2d(in_channels=194,out_channels=64,kernel_size=3,padding=1,stride=1,bias=True)

        self.add_conv_4 = nn.Conv2d(in_channels=258,out_channels=64,kernel_size=3,padding=1,stride=1,bias=True)
        self.add_conv_5 = nn.Conv2d(in_channels=322,out_channels=64,kernel_size=3,padding=1,stride=1,bias=True)
        self.add_conv_6 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1,bias=True)


        self.add_conv_middle = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1,bias=True)

    def forward(self, x):
        input = x
        #ada_conv = ada_channel_conv()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        # x = self.relu(x)
        x = torch.cat((x,input),1)
        #print('torch cat shape:',x.shape)
        for i in range(4):
            #print('i: {:02d}'.format(i))
            input = x
            x = self.relu(x)
            for j in range(3):
                #x = ada_conv(x)
                if i==0:
                    if i==0 and j==0:
                        x = self.add_conv_1(x)
                    else:
                        x = self.add_conv_middle(x)
                        #print('j : {:02d}'.format(j),x.shape)
                if i==1:
                    if i==1 and j==0:
                        x = self.add_conv_2(x)
                    else:
                        x = self.add_conv_middle(x)
                        #print('j : {:02d}'.format(j),x.shape)
                if i ==2:
                    if i==2 and j==0:
                        x = self.add_conv_3(x)
                    else:
                        x = self.add_conv_middle(x)
                        #print('j : {:02d}'.format(j),x.shape)
                if i==3:
                    if i==3 and j==0:
                        x = self.add_conv_4(x)
                    else:
                        x = self.add_conv_middle(x)
                        #print('j : {:02d}'.format(j),x.shape)
                if j < 2:
                    x = self.relu(x)
            x = torch.cat((x,input),1)
            #print('i : {:02d}'.format(i),x.shape)
            # x = self.relu(x)

        x = self.relu(x)
        x = self.add_conv_5(x)  # [batch channel nx ny]
        x = self.relu(x)
        x = self.conv4(x)


        return x

# if __name__ == '__main__':
#     model = net1x1().cuda()
#     num_params = 0
#     for param in model.parameters():
#         num_params += param.numel()
#     print(num_params)
#     x = torch.zeros((1, 2, 224, 224)).cuda()
#     y = model(x)
#     print(y.shape)