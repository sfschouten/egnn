import torch
import torch.nn as nn
import torch.nn.functional as F

class EGNN(nn.Module):
    
    def __init__(N, F_in, F_out, P):
        self.g = nn.Linear(F_in, F_out, bias=False)
        self.a = nn.Parameter(torch.empty(2 * F_out))
        
        self.N = N
        self.F_in = F_in
        self.F_out = F_out
        self.P = P

    def ds_normalize(self, E):
        """
        """
        E = E / E.sum(dim=1, keepdim=True)
        E = (E * E.permute(1, 0, 2) / E.sum(dim=0)).sum(dim=1)
        return E


    def attn_f(self, X):
        """
        X nodes features [N x F_in]
        """
        N,F = X.shape
        X = self.g(X) # N x F_out

        # N x N x F_out
        X_i = X.unsqueeze(0).repeat(N,N,F)
        X_j = X.unsqueeze(1).repeat(N,N,F)

        b = torch.cat(X_i, X_j, dim=2)      # N x N x 2*F_out
        return F.leaky_relu(b @ a).exp()    # N x N 


    def attn(self, X, E)
        """
        X nodes features [N x F_in]
        E edge features [N x N x F_in]
        """
        alpha = self.attn_f(X).unsqueeze(2) # N x N x 1
        alpha = alpha * E                   # N x N x P     
        alpha = ds_normalize(alpha)         
        return alpha


    def forward(self, X, E):
        """
        X nodes features [N x F_in]
        E edge features [N x N x F_in]

        """
        
        g = self.g(X)                   # N x F_out
        a = self.attn(X, E)             # N x N x P

        result = torch.matmul(a, g)     # N x F_out x P  
        result = result.view(self.N, -1)

        return result
