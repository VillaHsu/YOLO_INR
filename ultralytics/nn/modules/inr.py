import math
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_
from torch.autograd import Variable


__all__ = (
    "INR",
    "CoordFuser",
    "CC_Decoder",
    "Transformer",
)

class INR(nn.Module):
    def __init__(self, c1, out_dim, hidden_dim: list = [256, 256, 256, 256]):
        super().__init__()
        
        self.c1 = 384
        self.out_dim = out_dim
        layers = []
        lastv = self.c1*2
        for hidden in hidden_dim:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        shape = x.shape[:3]
        x = self.layers(x.contiguous().view(-1, x.shape[-1]))
        return x.view(*shape, -1).permute(0, 3, 1, 2)

class CoordFuser(nn.Module):
    def __init__(self, c1, sigma=10, m=64):
        super().__init__()
        self.sigma = 10
        self.m = 16

    def forward(self, x):
        batch_size, channels, imgsz = x.shape[:3]
        raw_coords = positional_encoding(get_coords([imgsz ,imgsz], x.device), self.sigma, self.m, batch_size)
        out = torch.cat([x, raw_coords], dim=1)
        return out

def get_coords(shape, device='gpu', ranges=None, flatten=False):
    """ 
    Make coordinates at grid centers.
    Args:
        shape   (list): image size [H, W]
        ranges  (list): grid boundaries [[left, right], [down, up]] 
        flatten (bool): True
    Returns:
        coords  (torch.tensor): H * W, 2
    """
    # determine the center of each grid
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -0.99999, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device).float()
        coord_seqs.append(seq)
    
    # make mesh
    coords = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        coords = coords.view(-1, coords.shape[-1])
    return coords

def positional_encoding(v: Tensor, sigma: float, m: int, batch_size: int) -> Tensor:
    """
    Args:
        v (Tensor): input tensor of shape :math:`(N, *, text{input_size})`
        sigma (float): constant chosen based upon the domain of :attr:`v`
        m (int): [description]
    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 dot m dot \text{input_size})`
    """
    j = torch.arange(m, device=v.device)
    coeffs = 2 * np.pi * sigma ** (j / m)
    vp = coeffs * torch.unsqueeze(v, -1)
    vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1).repeat(batch_size,1,1,1).permute(0, 3, 1, 2)

#print(positional_encoding(get_coords([512, 512], 'cpu'), 10, 64, 2).size())
#import torch
#print(torch.__version__)

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, channel):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        self.channel = channel

        
        grid_size = [16, 16] #for 64 [2,2] for 512 [16, 16]
        patch_size = ([img_size[0] // (img_size[0]//16) // grid_size[0], img_size[1] // (img_size[0]//16) // grid_size[1]]) # 512/32/16=1 (1,1)
        patch_size_real = (patch_size[0] * (img_size[0]//16), patch_size[1] * (img_size[0]//16)) # (32,32)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1]) # (16*16) = 256
        
        self.patch_embeddings = nn.Conv2d(in_channels=self.channel,
                                          out_channels=int(self.channel*1.5),
                                          kernel_size=patch_size,
                                          stride=patch_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, int(self.channel*1.5)))

        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        #print(x.size())
        features = x
        x = self.patch_embeddings(x) # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2) # (B, hidden, WH/P^2)
        x = x.transpose(-1, -2) # (B, n_patches, hidden)
       
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings, features

class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.out_channels = hidden_size
        self.fc1 = nn.Linear(self.out_channels, self.out_channels*4)
        self.fc2 = nn.Linear(self.out_channels*4, self.out_channels)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, vis, channel):
        super(Block, self).__init__()
        self.channel = channel
        self.hidden_size = int(self.channel*1.5)
        self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(self.hidden_size, 12, batch_first=True)
        self.ffn = Mlp(self.hidden_size)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x, x, x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, vis, channel):
        super(Encoder, self).__init__()
        self.channel = channel
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(int(self.channel*1.5), eps=1e-6)
        for _ in range(12):
            layer = Block(vis, channel=self.channel)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, c1, c2, img_size, vis=False):
        super(Transformer, self).__init__()
        self.c2 = c2
        self.img_size = img_size
        self.embeddings = Embeddings(img_size=self.img_size, channel=self.c2)
        self.encoder = Encoder(vis, channel=self.c2)
        self.conv_more = nn.Sequential(nn.Conv2d(int(self.c2*1.5), self.c2, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(self.c2),
                                       nn.SiLU(inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(int(self.c2*2), self.c2, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(self.c2),
                                       nn.SiLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(self.c2),
                                       nn.SiLU(inplace=True))

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output) # (B, n_patch, hidden)
        
        B, n_patch, hidden = encoded.size() # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = encoded.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        x = torch.cat([x, features], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        

        return x #, attn_weights, features











class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.Tanh):
        super(ResidLinear, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.linear(x) + x)

class CC_Decoder(nn.Module):
    def __init__(self, c1, c2, feature_size, pos_out_dim, imgsz):
        super(CC_Decoder, self).__init__()
        #self.m_size = m_size
        self.c1 = c1
        self.c2 = c2
        self.imgsz = imgsz
        
        self.n_features = int(feature_size*feature_size)
        self.pos_dim = pos_out_dim
        
        self.weight_dim = self.pos_dim + int(self.c1/self.pos_dim-1)*self.n_features + int(self.c1/self.pos_dim)
        
        self.res1 = ResidLinear(self.n_features, self.n_features)
        self.res2 = ResidLinear(self.n_features, self.n_features)
        self.res3 = ResidLinear(self.n_features, self.n_features)

        self.last1 = nn.Linear(self.n_features, self.c2)
        self.last2 = torch.nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.PReLU(),
            nn.Linear(self.n_features, 1),
            )
        
        self.act = nn.SiLU()
        
        self.act1 = nn.PReLU()#nn.Tanh()#nn.LeakyReLU()
        self.act2 = nn.PReLU()#nn.Tanh()#nn.LeakyReLU()
        self.act3 = nn.PReLU()#nn.Tanh()#nn.LeakyReLU()
        self.act4 = nn.PReLU()#nn.Tanh()#nn.LeakyReLU()
        self.act5 = nn.LeakyReLU()
        
        self.act6 = nn.LeakyReLU()
        
        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        
        #self.learn_prior = Variable(torch.rand(1).type(torch.FloatTensor), requires_grad=True).cuda()
        
        self.W_fine = nn.Linear(self.n_features, self.n_features)
        #self.W_fine2 = nn.Linear(self.weight_dim, self.weight_dim)
        
        self._initialize_weights() 
        self.omega_0 = 30.0
        self.kl = 0

        self.conv = nn.Conv2d(self.c1, self.c1+int(self.c1/self.n_features), kernel_size=1)

        self.sigma = 10
        self.m = int(int(feature_size*feature_size) / 4)
        

    def forward(self, x):### x1:z x2:coordinates
        batch_size, channels, imgsz = x.shape[:3]
        raw_coords = positional_encoding(get_coords([self.imgsz, self.imgsz], x.device), self.sigma, self.m, batch_size)
        x2 = raw_coords.permute(0, 2, 3, 1)
        
        x2 = x2.view(x2.shape[0], x2.shape[1]*x2.shape[1], x2.shape[-1])
        
        x1 = x
        
        x1 = self.conv(x1)
        
        b, n_query_pts = x2.shape[0], x2.shape[1] #x1 [b 1 8 8 ]x2 [b 1024 2*2*m]
        
        W = torch.reshape(x1, (b, self.weight_dim, self.n_features))
        W = self.W_fine(W)
        
        weight = []
        bias = []
        for n in range(int(self.c1/self.pos_dim)):
            weight.append(W[:, self.n_features*n+n:self.n_features*(n+1)+n, :])
            bias.append(W[:, self.n_features*(n+1)+n:self.n_features*(n+1)+n+1, :].repeat(1, n_query_pts, 1))

        #W1 = W[:,:self.pos_dim,:] 
        #b1 = W[:,self.pos_dim:self.pos_dim+1,:].repeat(1, n_query_pts, 1)
        
        
        #W2 = W[:,(self.pos_dim+1):(self.pos_dim+self.n_features+1),:]
        #b2 = W[:,(self.pos_dim+self.n_features+1):(self.pos_dim+self.n_features+2),:].repeat(1, n_query_pts, 1)
        #print(W1.shape, W2.shape,b1.shape,b2.shape)
        
        #W3 = W[:,(self.pos_dim+self.n_features+2):(self.pos_dim+2*self.n_features+2),:]
        #b3 = W[:,(self.pos_dim+2*self.n_features+2):(self.pos_dim+2*self.n_features+3),:].repeat(1, n_query_pts, 1)
        
        #W4 = W[:,(self.pos_dim+2*self.n_features+3):(self.pos_dim+3*self.n_features+3),:]
        #b4 = W[:,(self.pos_dim+3*self.n_features+3):(self.pos_dim+3*self.n_features+4),:].repeat(1, n_query_pts, 1)
        
        out = x2
        for j in range(int(self.c1/self.pos_dim)):
            out = self.act1(torch.einsum("bij, bjk -> bik", out, weight[j]) + bias[j])

        
        
        #out1 = torch.einsum("bij, bjk -> bik", x2, W1) + b1  ##[b 1024 self.n_features]
        #out1 = self.act1(out1)
   
        #out2 = torch.einsum("bij, bjk -> bik", out1, W2) + b2
        #out2 = self.act2(out2)+ out1

        #out3 = torch.einsum("bij, bjk -> bik", out2, W3) + b3
        #out3 = self.act3(out3) + out2
   
        #out4 = torch.einsum("bij, bjk -> bik", out3, W4) + b4
        #out4 = self.act4(out4) + out3
        
        
        out = self.act(self.last1(out))#torch.exp(torch.squeeze(self.last1(out4)))
        #out_sigma = torch.exp(torch.squeeze(self.last2(out4)))
        
        #self.kl = torch.mean((0.5*out_sigma**2 + 0.5*(out_mu)**2 - torch.log(out_sigma) - 1/2))
        out = out.permute(0, 2, 1)
        out = out.reshape([b, self.c2, self.imgsz, self.imgsz])    

        return out #, out_sigma, self.kl
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                #nn.init.kaiming_uniform_(m.weight)
                #nn.init.constant_(m.weight, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 
                    #nn.init.kaiming_uniform_(m.bias)
