import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_dense_batch
import numpy as np
import sys
sys.path.append('/data/conghao001/pharmacophore2drug/PP2Drug/src')
from data_processing.utils import make_edge_mask

def compute_distance(x, edge_index):
    row, col = edge_index
    distance = x[row] - x[col]
    radial = torch.sum((distance) ** 2, dim=1, keepdim=True)
    norm = torch.sqrt(radial + 1e-8)
    distance = distance/(norm + 1)
    return radial, distance


def compose_graph(h0, h1, x0, x1, node_mask, edge_mask, batch, max_N=86):
    '''
        first graph is G_t, and second graph is G_T
        node mask and edge mask for G_t and G_T are the same, so we only need to pass one of them. Same for batch
        (Bs, N_node, Feat_size)
        integrate over the N_node dimension

        Output:
            h: (Bs, 2*max_N, Feat_size)
            x: (Bs, 2*max_N, Feat_size)
            node_mask: (Bs, 2*max_N)  # keep G_t and G_T nodes
            edge_mask: (Bs, 2*max_N * 2*max_N)
            node_mask_Gt: (Bs, 2*max_N) # only keep G_t nodes
    '''
    batch_size = h0.size(0)
    new_h_ctr = torch.zeros([batch_size, 2*max_N, h0.size(-1)], device=h0.device)
    new_x_ctr = torch.zeros([batch_size, 2*max_N, x0.size(-1)], device=h0.device)
    new_node_mask_ctr = torch.zeros([batch_size, 2*max_N], device=h0.device).bool()
    node_mask_Gt = torch.zeros([batch_size, 2*max_N], device=h0.device).bool()
    new_edge_mask_list = []
    for i in range(batch_size):
        n_node = node_mask[i].sum()
        new_h_ctr[i, :n_node, ] = h0[i, :n_node, ]
        new_h_ctr[i, n_node:2*n_node, ] = h1[i, :n_node, ]
        new_x_ctr[i, :n_node, ] = x0[i, :n_node, ]
        new_x_ctr[i, n_node:2*n_node, ] = x1[i, :n_node, ]

        new_node_mask_ctr[i, :2*n_node] = True
        node_mask_Gt[i, :n_node] = True
        edge_mask = make_edge_mask(2*n_node, 2*max_N, device=h0.device)
        new_edge_mask_list.append(edge_mask)
        
        # print(n_node)
        
    # new_edge_mask_ctr = torch.tensor(np.array(new_edge_mask_list))
    new_edge_mask_ctr = torch.cat(new_edge_mask_list, dim=0)

    return new_h_ctr, new_x_ctr, new_node_mask_ctr, new_edge_mask_ctr, node_mask_Gt


def compose_graph_wrong(h0, h1, x0, x1, node_mask, edge_mask, batch, max_N=86):
    '''
        first graph is G_t, and second graph is G_T
        node mask and edge mask for G_t and G_T are the same, so we only need to pass one of them. Same for batch
        (Bs, N_node, Feat_size)
        integrate over the N_node dimension
    '''
    batch_size = h0.size(0)
    N = batch.size(0)   # real number of nodes in all graphs in the batch
    h0 = h0.view(batch_size*max_N, -1)
    h1 = h1.view(batch_size*max_N, -1)
    x0 = x0.view(batch_size*max_N, -1)
    x1 = x1.view(batch_size*max_N, -1)
    # node_mask = node_mask.view(batch_size*max_N, -1)

    # sparse_h0 = h0 * node_mask
    # sparse_h1 = h1 * node_mask
    # sparse_x0 = x0 * node_mask
    # sparse_x1 = x1 * node_mask
    dense_batch, batch_mask = to_dense_batch(batch, max_num_nodes=max_N*batch_size, fill_value=batch_size)
    dense_batch = dense_batch[0]
    batch_merged = torch.cat([dense_batch, dense_batch], dim=0)
    sort_idx = torch.sort(batch_merged, stable=True).indices

    node_mask_1d = node_mask.view(batch_size*max_N)
    mask_G_t = torch.cat([
        node_mask_1d, 
        torch.zeros([batch_size*max_N], device=h0.device).bool(),
    ], dim=0)
    mask_G_t = mask_G_t[sort_idx] #.view(batch_size, 2*max_N)
    # mask_G_t = torch.cat([
    #     node_mask, 
    #     torch.zeros([batch_size, max_N], device=h0.device).bool(),
    # ], dim=0)
    # print(mask_G_t.size())
    # mask_G_t = mask_G_t.view(batch_size*2*max_N, -1)[sort_idx].view(batch_size, 2*max_N)
    h = torch.cat([h0, h1], dim=0)
    x = torch.cat([x0, x1], dim=0)
    h = h[sort_idx]
    x = x[sort_idx]
    # h = h.view(batch_size*2*max_N, -1)[sort_idx].view(batch_size, 2*max_N, -1)
    # x = x.view(batch_size*2*max_N, -1)[sort_idx].view(batch_size, 2*max_N, -1)

    return h, x, mask_G_t


class EGCL(nn.Module):   # Graph Convolutional Layer
    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()

        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
         
        self.att_mlp = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Sigmoid())

    def forward(self, h, edge_index, edge_attr, flags, edge_mask):
        row, col = edge_index

        # Message
        msg = torch.cat([h[row], h[col], edge_attr], dim=1)
        msg = self.msg_mlp(msg)

        att = self.att_mlp(msg)
        msg = (msg * att) * edge_mask

        # Aggregation
        agg = unsorted_segment_sum(msg, row, num_segments=h.size(0),
                                normalization_factor=1,
                                aggregation_method='sum')

        agg = torch.cat([h, agg], dim=1)
        h = h + self.node_mlp(agg)
        return h * flags

class E3CoordLayer(nn.Module):
    def __init__(self, hidden_dim, coords_range, edge_dim):
        super().__init__()
        self.tanh = True
        self.coords_range = coords_range

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        torch.nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

    def forward(self, h, x, edge_index, edge_attr, distance, flags, edge_mask):
        row, col = edge_index

        msg = torch.cat([h[row], h[col], edge_attr], dim=1)
        trans = distance * torch.tanh(self.coord_mlp(msg)) * self.coords_range
        # tanns = trans * edge_mask    # should be a type??
        trans = trans * edge_mask

        agg = unsorted_segment_sum(trans, row, num_segments=x.size(0),
                                normalization_factor=1,
                                aggregation_method='sum')
        x = x + agg
        return x * flags

class E3Block(nn.Module):
    def __init__(self, nhid, coords_range, n_layers=2):
        super().__init__()
        edge_dim = 2
        self.coords_range = coords_range
        self.gcl = nn.ModuleList([])
        for _ in range(n_layers):
            gcl = EGCL(nhid, nhid, edge_dim)
            self.gcl.append(gcl)

        self.e3_coord_layer = E3CoordLayer(nhid, coords_range, edge_dim)

    def forward(self, h, x, edge_index, d, flags, edge_mask):
        d_, distance = compute_distance(x, edge_index)
        edge_attr = torch.cat([d_, d], dim=1)
        for gcl in self.gcl:
            h = gcl(h, edge_index, edge_attr, flags, edge_mask)
        x = self.e3_coord_layer(h, x, edge_index, edge_attr, distance, flags, edge_mask)

        return h, x

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fixed_offset=True):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        if fixed_offset:
            # customized offset
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
        else:
            offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AngleExpansion(nn.Module):
    def __init__(self, start=1.0, stop=5.0, half_expansion=10):
        super(AngleExpansion, self).__init__()
        l_mul = 1. / torch.linspace(stop, start, half_expansion)
        r_mul = torch.linspace(start, stop, half_expansion)
        coeff = torch.cat([l_mul, r_mul], dim=-1)
        self.register_buffer('coeff', coeff)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.coeff.view(1, -1))


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    'silu': nn.SiLU()
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False, verbose=False):
        super().__init__()
        self.verbose = verbose
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    # layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)
        # print('MLP structure:', self.net)

    def forward(self, x):
        if self.verbose:
            print('input:', x)
            for layer in self.net:
                x = layer(x)
                print('apply layer', layer)
                if isinstance(layer, nn.Linear):
                    print('layer weights:', layer.state_dict()['weight'])
                    print('layer bias:', layer.state_dict()['bias'])
                print('output:', x)
            return x
        else:
            return self.net(x)


def outer_product(*vectors):
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()


def get_h_dist(dist_metric, hi, hj):
    if dist_metric == 'euclidean':
        h_dist = torch.sum((hi - hj) ** 2, -1, keepdim=True)
        return h_dist
    elif dist_metric == 'cos_sim':
        hi_norm = torch.norm(hi, p=2, dim=-1, keepdim=True)
        hj_norm = torch.norm(hj, p=2, dim=-1, keepdim=True)
        h_dist = torch.sum(hi * hj, -1, keepdim=True) / (hi_norm * hj_norm)
        return h_dist, hj_norm


def get_r_feat(r, r_exp_func, node_type=None, edge_index=None, mode='basic'):
    if mode == 'origin':
        r_feat = r
    elif mode == 'basic':
        r_feat = r_exp_func(r)
    elif mode == 'sparse':
        src, dst = edge_index
        nt_src = node_type[src]  # [n_edges, 8]
        nt_dst = node_type[dst]
        r_exp = r_exp_func(r)
        r_feat = outer_product(nt_src, nt_dst, r_exp)
    else:
        raise ValueError(mode)
    return r_feat


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    # previous version has problems when ligand atom types are fixed
    # (due to sorting randomly in case of same element)

    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    # sort_idx = batch_ctx.argsort()
    sort_idx = torch.sort(batch_ctx, stable=True).indices

    mask_ligand = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx, mask_ligand


def compose_context_prop(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    sort_idx = batch_ctx.argsort()

    mask_protein = torch.cat([
        torch.ones([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.zeros([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]       # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


def hybrid_edge_connection(ligand_pos, protein_pos, k, ligand_index, protein_index):
    # fully-connected for ligand atoms
    dst = torch.repeat_interleave(ligand_index, len(ligand_index))
    src = ligand_index.repeat(len(ligand_index))
    mask = dst != src
    dst, src = dst[mask], src[mask]
    ll_edge_index = torch.stack([src, dst])

    # knn for ligand-protein edges
    ligand_protein_pos_dist = torch.unsqueeze(ligand_pos, 1) - torch.unsqueeze(protein_pos, 0)
    ligand_protein_pos_dist = torch.norm(ligand_protein_pos_dist, p=2, dim=-1)
    knn_p_idx = torch.topk(ligand_protein_pos_dist, k=k, largest=False, dim=1).indices
    knn_p_idx = protein_index[knn_p_idx]
    knn_l_idx = torch.unsqueeze(ligand_index, 1)
    knn_l_idx = knn_l_idx.repeat(1, k)
    pl_edge_index = torch.stack([knn_p_idx, knn_l_idx], dim=0)
    pl_edge_index = pl_edge_index.view(2, -1)
    return ll_edge_index, pl_edge_index


def batch_hybrid_edge_connection(x, k, mask_ligand, batch, add_p_index=False):
    batch_size = batch.max().item() + 1
    batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index = [], [], []
    with torch.no_grad():
        for i in range(batch_size):
            ligand_index = ((batch == i) & (mask_ligand == 1)).nonzero()[:, 0]
            protein_index = ((batch == i) & (mask_ligand == 0)).nonzero()[:, 0]
            ligand_pos, protein_pos = x[ligand_index], x[protein_index]
            ll_edge_index, pl_edge_index = hybrid_edge_connection(
                ligand_pos, protein_pos, k, ligand_index, protein_index)
            batch_ll_edge_index.append(ll_edge_index)
            batch_pl_edge_index.append(pl_edge_index)
            if add_p_index:
                all_pos = torch.cat([protein_pos, ligand_pos], 0)
                p_edge_index = knn_graph(all_pos, k=k, flow='source_to_target')
                p_edge_index = p_edge_index[:, p_edge_index[1] < len(protein_pos)]
                p_src, p_dst = p_edge_index
                all_index = torch.cat([protein_index, ligand_index], 0)
                p_edge_index = torch.stack([all_index[p_src], all_index[p_dst]], 0)
                batch_p_edge_index.append(p_edge_index)

    if add_p_index:
        edge_index = [torch.cat([ll, pl, p], -1) for ll, pl, p in zip(
            batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index)]
    else:
        edge_index = [torch.cat([ll, pl], -1) for ll, pl in zip(batch_ll_edge_index, batch_pl_edge_index)]
    edge_index = torch.cat(edge_index, -1)
    return edge_index
