"""Deep matrix factorization with explicit, signed row/column interactions."""
import torch
from torch import nn


class BilinearDMF(nn.Module):
    """Low-rank factors, residual neural towers, biases and a latent mask head.

    Intensities remain a row/column inner product. The auxiliary classifier
    shares row factors without forcing reconstructed intensities to encode the
    binary input mask directly. No held-out intensity is an input to this class.
    """
    architecture_version = 'bilinear-residual-v1'

    def __init__(self,n_rows,n_cols,embedding_dim=64,hidden_dims=None,mask_predictor_hidden_dim=128,interaction_dim=None):
        super().__init__()
        self.n_rows,self.n_cols,self.embedding_dim=n_rows,n_cols,embedding_dim
        self.interaction_dim=min(8,embedding_dim) if interaction_dim is None else interaction_dim
        if type(self.interaction_dim) is not int or not 1<=self.interaction_dim<=embedding_dim:
            raise ValueError('DMF interaction rank must be between 1 and latent dimension')
        self.row_embedding=nn.Embedding(n_rows,embedding_dim)
        self.col_embedding=nn.Embedding(n_cols,embedding_dim)
        self.row_bias=nn.Embedding(n_rows,1)
        self.col_bias=nn.Embedding(n_cols,1)
        hidden_dims=[256,128] if hidden_dims is None else hidden_dims
        def tower():
            layers=[];width=embedding_dim
            for hidden in hidden_dims:
                layers.extend([nn.Linear(width,hidden),nn.GELU(),nn.Dropout(.1)])
                width=hidden
            layers.append(nn.Linear(width,self.interaction_dim))
            return nn.Sequential(*layers)
        self.row_tower,self.col_tower=tower(),tower()
        width=mask_predictor_hidden_dim
        self.mask_predictor=nn.Sequential(nn.Linear(embedding_dim,width),nn.GELU(),
            nn.Linear(width,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,n_cols))
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        for embedding in (self.row_embedding,self.col_embedding):
            nn.init.normal_(embedding.weight,std=.05)
        for bias in (self.row_bias,self.col_bias):nn.init.zeros_(bias.weight)
        for network in (self.row_tower,self.col_tower):
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
        self.initialization={'method':'small random factors','rank':self.interaction_dim}

    @torch.no_grad()
    def initialize_from_observed(self,values,input_mask):
        """Warm start from the fixed visible training input, never its holdouts."""
        if tuple(values.shape)!=(self.n_rows,self.n_cols) or input_mask.shape!=values.shape:
            raise ValueError('DMF initialization input shape does not match factors')
        visible=input_mask.to(device=self.row_embedding.weight.device,dtype=torch.bool)
        values=values.to(device=visible.device,dtype=self.row_embedding.weight.dtype)
        if not torch.isfinite(values[visible]).all():
            raise ValueError('DMF initialization requires finite visible training values')
        clean=torch.where(visible,values,0.)
        means=clean.sum(dim=1)/visible.sum(dim=1).clamp_min(1)
        self.row_bias.weight.copy_(means[:,None])
        residual=torch.where(visible,clean-means[:,None],0.)
        rank=min(self.interaction_dim,self.n_rows,self.n_cols)
        if min(residual.shape)<=self.interaction_dim+8:
            left,singular,right_t=torch.linalg.svd(residual,full_matrices=False)
            right=right_t.T
        else:
            left,singular,right=torch.svd_lowrank(residual,q=min(rank+8,min(residual.shape)),niter=3)
        root=singular[:rank].clamp_min(0).sqrt()
        self.row_embedding.weight[:,:rank].copy_(left[:,:rank]*root)
        self.col_embedding.weight[:,:rank].copy_(right[:,:rank]*root)
        self.initialization={'method':'visible-training-only truncated SVD','rank':rank,
            'observations':int(visible.sum())}
        return dict(self.initialization)

    def _row_factors(self,indices):
        factors=self.row_embedding(indices)
        return factors[:,:self.interaction_dim]+self.row_tower(factors)

    def _col_factors(self,indices=None):
        factors=self.col_embedding.weight if indices is None else self.col_embedding(indices)
        return factors[:,:self.interaction_dim]+self.col_tower(factors)

    def _reconstruct_rows(self,row_indices):
        return (self._row_factors(row_indices)@self._col_factors().T
            +self.row_bias(row_indices)+self.col_bias.weight.T)

    def _forward_indices(self,row_indices,col_indices):
        return ((self._row_factors(row_indices)*self._col_factors(col_indices)).sum(dim=-1)
            +self.row_bias(row_indices).squeeze(-1)+self.col_bias(col_indices).squeeze(-1))

    def forward(self,x,row_indices=None):
        if isinstance(x,tuple) and len(x)==2:
            rows,columns=x
            return self._forward_indices(rows,columns),self.mask_predictor(self.row_embedding(rows))
        if x.ndim!=2:raise ValueError('DMF input must have shape (batch, proteins)')
        if row_indices is None:row_indices=torch.arange(len(x),device=x.device)
        if row_indices.ndim!=1 or len(row_indices)!=len(x):
            raise ValueError('row_indices must contain one sample index per batch row')
        factors=self._row_factors(row_indices)
        reconstruction=factors@self._col_factors().T+self.row_bias(row_indices)+self.col_bias.weight.T
        return reconstruction,self.mask_predictor(self.row_embedding(row_indices))

    def get_full_matrix(self):
        return self._reconstruct_rows(torch.arange(self.n_rows,device=self.row_embedding.weight.device))


def build_dmf(architecture,**kwargs):
    if architecture=='bilinear':
        requested=kwargs.get('interaction_dim')
        if requested is not None and (type(requested) is not int or requested<1):
            raise ValueError('DMF interaction rank must be a positive integer')
        kwargs['interaction_dim']=min(8 if requested is None else requested,kwargs.get('embedding_dim',64))
        return BilinearDMF(**kwargs)
    if architecture=='legacy':
        kwargs.pop('interaction_dim',None)
        from ._native.models import LegacyDMF
        return LegacyDMF(**kwargs)
    raise ValueError('DMF architecture must be bilinear or legacy')
