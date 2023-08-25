import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing

def interpolate(phi, edge_index, interp_weights):
    print('Interpolate')
    node_nei = edge_index[0,:]
    node_own = edge_index[1,:]

    phi_nei = phi[node_nei,:]
    phi_own = phi[node_own,:]

    phi_f = interp_weights * (phi_own - phi_nei) + phi_nei
    
    return phi_f 

def gradient(phi, edge_index, interp_weights, 
             sf, edge_sign, surface_integrator):
    """ 
    GRAD(phi) = \sum_f phi_f * S_f * n_f [INCREASES DIMENSIONALITY]
    """
    print('Computing gradient.')
    phi_f = interpolate(phi, edge_index, interp_weights)

    # expand phi_f
    phi_f = phi_f[:,:,None]
    sf_sign = sf * edge_sign
    sf_sign = sf_sign[:, None, :]
    edge_attr = phi_f * sf_sign

    # Reshape edge attr 
    n_edges = edge_attr.shape[0]
    n_features = edge_attr.shape[1]
    n_spatial_dims = edge_attr.shape[2]
    edge_attr = edge_attr.view((n_edges, n_features * n_spatial_dims))

    # Evaluate gradient 
    gradient = surface_integrator(phi, edge_index, edge_attr) # phi is a dummy here

    # reshape gradient 
    grad_phi = gradient.view((-1, n_features, n_spatial_dims))

    return grad_phi

def divergence(phi, edge_index, interp_weights, 
               sf, edge_sign, surface_integrator):
    """ 
    DIV(phi) = \sum_f dot(phi_f, S_f * n_f) [DECREASES DIMENSIONALITY]
    """
    phi_f = interpolate(phi, edge_index, interp_weights)
    sf_sign = sf * edge_sign

    # Edge-wise dot product 
    edge_attr = torch.einsum('ij,ij->i', phi_f, sf_sign)
    edge_attr = edge_attr[:, None]

    # evaluate divergence 
    div_phi = surface_integrator(phi, edge_index, edge_attr) # phi is a dummy here
    
    return div_phi


def laplacian(phi, pos, edge_index, interp_weights, 
               sf, edge_sign, surface_integrator):
    """ 
    LAP(phi) = \sum_f |S_f| * grad(phi_f) [DOES NOT CHANGE DIMENSIONALITY]
    """
    phi_f = interpolate(phi, edge_index, interp_weights)

    node_nei = edge_index[0,:]
    node_own = edge_index[1,:]

    phi_nei = phi[node_nei,:]
    phi_own = phi[node_own,:]

    pos_nei = pos[node_nei,:]
    pos_own = pos[node_own,:]

    # Compute face normal aligned gradient
    grad_f = (phi_nei - phi_own)/(torch.norm(pos_nei - pos_own, dim = 1, keepdim=True))
    sf_sign = sf * edge_sign
    mag_sf_sign = torch.norm(sf_sign, dim=1, keepdim=True)

    # Scale by face area 
    edge_attr = grad_f * mag_sf_sign

    # Evaluate laplacian
    lap = surface_integrator(phi, edge_index, edge_attr) # phi is a dummy here
    return lap


class EdgeAggregation(MessagePassing):
    r"""This is a custom class that returns node quantities that represent the neighborhood-averaged edge features.
    Args:
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or 
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`, 
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    propagate_type = {'x': Tensor, 'edge_attr': Tensor}

    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        x_j = edge_attr
        return x_j

    def __repr__(self) -> str: 
        return f'{self.__class__.__name__}'



