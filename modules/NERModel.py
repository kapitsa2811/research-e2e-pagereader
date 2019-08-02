import torch.nn as nn

class NERModel(nn.Module):
    # Module to calculate character probabilities for roi-pool output feature map

    def __init__(self,feature_size = 256,pool_h=32,n_classes=4):
        super(NERModel,self).__init__()
        self.n_classes=n_classes
        pool_w=280

        nh= 256
        self.conv1 = nn.Conv2d(feature_size, nh, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
    
        self.conv2 = nn.Conv2d(nh, nh, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        
        '''self.rnn = nn.Sequential(
        BidirectionalLSTM(pool_h*nh, nh,nh),
        BidirectionalLSTM(nh,nh, pool_h*nh))'''

        self.output = nn.Linear(in_features=pool_h*nh*pool_w,out_features=self.n_classes)
        
    def forward(self,x):
        x= self.conv1(x)
        x = self.act1(x)
    
        x = self.conv2(x)
        x = self.act2(x)
    
        x=x.view(x.shape[0],1,x.shape[1]*x.shape[2]*x.shape[3])
    
        #x = self.rnn(x)
        output = self.output(x)
        return output
