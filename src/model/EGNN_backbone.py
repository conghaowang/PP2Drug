import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_sum
from torch_geometric.nn import radius_graph, knn_graph

from model.utils.utils_EGNN import E3Block, E3CoordLayer, compute_distance, compose_graph, GaussianSmearing, MLP, batch_hybrid_edge_connection, NONLINEARITIES

class EGNN(torch.nn.Module):
    def __init__(self, max_feat_num, coord_dim, n_layers, hidden_size, max_node_num, include_charge=False, time_cond=True, xT_mode='concat_graph', data_mode='single'):
        super().__init__()
        self.nfeat = max_feat_num + int(include_charge)
        self.coord_dim = coord_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.time_cond = time_cond
        self.include_charge = include_charge
        self.max_N = max_node_num
        self.xT_mode = xT_mode
        self.data_mode = data_mode

        # Fully connected edge index
        adj = torch.ones((self.max_N, self.max_N))
        self.full_edge_index = adj.nonzero(as_tuple=False).T

        self.embedding_in = nn.Linear(self.nfeat + int(self.time_cond), self.hidden_size)
        self.embedding_out = nn.Linear(self.hidden_size, self.nfeat)

        self.layers = torch.nn.ModuleList()
        coords_range = float(15 / self.n_layers)
        for _ in range(self.n_layers):
            self.layers.append(E3Block(self.hidden_size, coords_range))

        if self.xT_mode == 'concat_latent' and self.data_mode == 'single':
            self.embedding_in_T = nn.Linear(self.nfeat, self.hidden_size)
            self.embedding_out_T = nn.Linear(self.hidden_size, self.nfeat)
            self.xT_layers = nn.ModuleList()
            self.x_ctr_layers = nn.ModuleList()
            for _ in range(self.n_layers):
                self.xT_layers.append(E3Block(self.hidden_size, coords_range))
                self.x_ctr_layers.append(E3CoordLayer(self.hidden_size, coords_range, 2))
            self.final_x_output = nn.Linear(self.coord_dim * 2, self.coord_dim)
            self.final_h_output =  nn.Linear(self.nfeat * 2, self.nfeat)
    
    def forward(self, h, x, hT, xT, t=None, batch_info=None, node_mask=None, edge_mask=None, Gt_mask=None, xT_mode='concat_graph'):
        '''
            h: one-hot encoded atom type
            x: coordinates
            flags: node mask (0 for padding)
            edge_mask: edge mask (0 for padding)
            batch_info: batch.batch
        '''
        # x_ = x
        bs, n_nodes_single_g, _ = h.shape
        # print(f'bs: {bs}, num of nodes in a single g: {n_nodes_single_g}')
        # h = h.view(bs * n_nodes, -1)
        # x = x.view(bs * n_nodes, -1)
        if self.data_mode == 'single':
            if self.xT_mode.startswith('concat'):
                if self.xT_mode == 'concat_graph':
                    ht, xt, new_node_mask, new_edge_mask, node_mask_Gt = compose_graph(h, hT, x, xT, node_mask, edge_mask, batch_info)
                    n_nodes = 2 * n_nodes_single_g
                elif self.xT_mode == 'concat_latent':
                    ht, xt = h, x
                    new_node_mask = node_mask
                    new_edge_mask = edge_mask
                    node_mask_Gt = node_mask
                    n_nodes = n_nodes_single_g
                    # masks are applied after Gt and GT passed the EGNN respectively, but they can be generated in advance
                    _, _, node_mask_ctr, edge_mask_ctr, node_mask_Gt_ctr = compose_graph(ht, hT, xt, xT, node_mask, edge_mask, batch_info)
                else:
                    raise ValueError('Invalid concatenation mode')
            elif self.xT_mode == 'none':
                ht, xt = h, x
                new_node_mask = node_mask
                new_edge_mask = edge_mask
                node_mask_Gt = node_mask
                n_nodes = n_nodes_single_g
            else:
                raise ValueError('Invalid xT_mode')
        elif self.data_mode == 'combined':
            assert self.xT_mode == 'none'
            assert Gt_mask is not None
            ht, xt = h, x
            new_node_mask = node_mask
            new_edge_mask = edge_mask
            node_mask_Gt = Gt_mask
            n_nodes = n_nodes_single_g
        else:
            raise ValueError('Invalid data_mode')
        
        x_  = xt

        # all the data is for the integrated graph G_t + G_T now
        ht = ht.view(bs * n_nodes, -1)
        xt = xt.view(bs * n_nodes, -1)
        # node_mask_Gt = node_mask_Gt.detach().clone()
        node_mask_Gt = node_mask_Gt.view(bs * n_nodes, -1)
        new_node_mask = new_node_mask.view(bs * n_nodes, -1)
        
        edge_index = self.make_edge_index(bs, n_nodes).to(ht.device)
        # new_edge_mask = new_edge_mask.detach().clone()
        new_edge_mask = new_edge_mask.view(bs * n_nodes * n_nodes, -1)

        d, _ = compute_distance(xt, edge_index)

        if self.time_cond:
            t = t.view(bs, 1).repeat(1, n_nodes).view(bs * n_nodes, 1)
            ht = torch.cat([ht, t], dim=1)

        ht = self.embedding_in(ht) * new_node_mask
        for layer in self.layers:
            ht, xt = layer(ht, xt, edge_index, d, new_node_mask, new_edge_mask)
            
        ht = self.embedding_out(ht) * node_mask_Gt
        xt = xt * node_mask_Gt

        if self.xT_mode == 'concat_latent':
            # not working for now
            # update GT alone
            hT = hT.view(bs * n_nodes, -1)
            xT = xT.view(bs * n_nodes, -1)

            d_T, _ = compute_distance(xT, edge_index)
            hT = self.embedding_in_T(hT) * new_node_mask
            for layer in self.xT_layers:
                hT, xT = layer(hT, xT, edge_index, d_T, new_node_mask, new_edge_mask)
            hT = self.embedding_out_T(hT) * new_node_mask
            xT = xT * new_node_mask

            # concatenate the xt and xT and apply E3CoordLayer
            edge_index_ctr = self.make_edge_index(bs, 2*n_nodes).to(ht.device)
            # _, _, node_mask_ctr, edge_mask_ctr, node_mask_Gt_ctr = compose_graph(ht, hT, xt, xT, node_mask, edge_mask, batch_info)

            ht_coord = torch.cat([ht, hT], dim=0)
            xt = torch.cat([xt, xT], dim=0)    # concatenate the xT to the xt along the bs*n_nodes dimension
            # xt = self.final_x_output(xt)    # use e3_coord layer to output the final x
            d_ctr, distance_ctr = compute_distance(xt, edge_index_ctr)
            edge_attr_ctr = torch.cat([d_ctr, d_ctr], dim=1)
            for layer in self.x_ctr_layers:
                xt = layer(ht_coord, xt, edge_index_ctr, edge_attr_ctr, distance_ctr, node_mask_Gt_ctr, edge_mask_ctr)

            # only keep the xt part
            xt = xt[:bs * n_nodes, :]

            # h is invariant, so just concatenate the hT to the ht and apply a linear layer to merge them
            ht = torch.cat([ht, hT], dim=-1)
            ht = self.final_h_output(ht) * node_mask_Gt
        
        if self.include_charge:
            charge = ht[:, -1:]
            ht = ht[:, :self.nfeat-1]
        
        ht = F.softmax(ht, dim=-1)

        ht = ht.view(bs, n_nodes, -1)
        xt = xt.view(bs, n_nodes, -1) #- x_    # why subtract x_?
        if self.include_charge:
            charge = charge.view(bs, n_nodes, -1)
            ht = torch.cat([ht, charge], dim=-1)
        
        # just keep the G_t part
        if self.xT_mode == 'concat_graph':
            ht = ht[: , :n_nodes_single_g, :]
            xt = xt[: , :n_nodes_single_g, :]
        return ht, xt

    def make_edge_index(self, bs, max_N=86*2):
        adj = torch.ones((max_N, max_N))
        full_edge_index = adj.nonzero(as_tuple=False).T
        edge_index = []
        for i in range(bs):
            edge_index.append(full_edge_index + (i * self.max_N))
        
        edge_index = torch.cat(edge_index, dim=1)

        return edge_index


class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn='silu', norm=False):
        super().__init__()
        self.r_min = 0.
        self.r_max = 10.
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        if num_r_gaussian > 1:
            self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)
        self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian, hidden_dim, hidden_dim,
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True, verbose=False)
        # print('edge MLP architecture:', self.edge_mlp)
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            # self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
            x_mlp = [nn.Linear(hidden_dim, hidden_dim), NONLINEARITIES[act_fn]]
            layer = nn.Linear(hidden_dim, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            x_mlp.append(layer)
            x_mlp.append(nn.Tanh())
            self.x_mlp = nn.Sequential(*x_mlp)

        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)

    def forward(self, h, x, edge_index, Gt_mask, edge_attr=None):
        # print('mask', Gt_mask)
        src, dst = edge_index
        hi, hj = h[dst], h[src]
        # \phi_e in Eq(3)
        rel_x = x[dst] - x[src]
        d_sq = torch.sum(rel_x ** 2, -1, keepdim=True)
        if self.num_r_gaussian > 1:
            d_feat = self.distance_expansion(torch.sqrt(d_sq + 1e-8))
        else:
            d_feat = d_sq
        if edge_attr is not None:
            edge_feat = torch.cat([d_feat, edge_attr], -1)
        else:
            edge_feat = d_sq

        # print('input to edge mlp')
        # print('hi', hi)
        # print('hj', hj)
        # print('edge_feat', edge_feat)
        mij = self.edge_mlp(torch.cat([hi, hj, edge_feat], -1))
        # print('mij', mij)
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])

        # h update in Eq(6)
        h = self.node_mlp(torch.cat([mi, h], -1)) * Gt_mask[:, None] + h # * (~Gt_mask)[:, None]
        # h = h * Gt_mask
        if self.update_x:
            # x update in Eq(4)
            xi, xj = x[dst], x[src]
            # (xi - xj) / (\|xi - xj\| + C) to make it more stable
            # print(xi - xj)
            # print('input to delta_x computation')
            # print('d_sq', d_sq)
            # print('mlp output', self.x_mlp(mij))
            delta_x = scatter_sum((xi - xj) / (torch.sqrt(d_sq + 1e-8) + 1) * self.x_mlp(mij), dst, dim=0)
            # print(x.size(), delta_x.size(), Gt_mask[:, None].size())
            # print(delta_x)
            x = delta_x * Gt_mask[:, None] + x  # only Gt positions will be updated
            # x = delta_x * Gt_mask + x  # only Gt positions will be updated
            # x = x + delta_x

        return h, x


class EGNN_combined_graph(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, edge_feat_dim, num_r_gaussian, time_cond=True, k=32, cutoff=10.0, cutoff_mode='knn',
                 update_x=True, act_fn='silu', norm=False):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.time_cond = time_cond
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        self.k = k
        self.cutoff = cutoff
        self.cutoff_mode = cutoff_mode
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_r_gaussian)
        if self.time_cond:
            self.embedding_in = nn.Linear(self.input_dim + 1, self.hidden_dim)
        else:
            self.embedding_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.embedding_out = nn.Linear(self.hidden_dim, self.input_dim)
        self.net = self._build_network()

    def _build_network(self):
        # Equivariant layers
        layers = []
        for l_idx in range(self.num_layers):
            layer = EnBaseLayer(self.hidden_dim, self.edge_feat_dim, self.num_r_gaussian,
                                update_x=self.update_x, act_fn=self.act_fn, norm=self.norm)
            layers.append(layer)
        return nn.ModuleList(layers)

    # todo: refactor
    def _connect_edge(self, x, Gt_mask, batch, bs, n_node):
        # if self.cutoff_mode == 'radius':
        #     edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        # bs, n_node = x.size(0), x.size(1)
        # print(bs, n_node)
        # batch_ = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        # print('build edge on x', x)
        # for i in range(bs):
        #     batch_[i * n_node:(i + 1) * n_node] = i
        if self.cutoff_mode == 'knn':
            # print(x.size(), batch.size())
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
            # print('edge index', edge_index)
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=Gt_mask, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    # todo: refactor
    @staticmethod
    def _build_edge_type(edge_index, Gt_mask):
        # print(edge_index.size())
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        # print(Gt_mask.size())   
        n_src = Gt_mask[src] == True # 1
        n_dst = Gt_mask[dst] == True # 1
        # print(n_src.size(), n_dst.size())
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, t, Gt_mask, batch, return_all=False):
        # bs, n_node = x.size(0), x.size(1)
        bs = batch.max().item() + 1
        # print('batch size', bs)
        # x = x.view(bs * n_node, -1)
        # h = h.view(bs * n_node, -1)
        n_node = Gt_mask.sum() * 2    # not used any more
        # print('input x:', x)
        # print('input h:', h)
        h_ = h

        if self.time_cond:
            t = t.unsqueeze(1)
            # print(h.size(), t.size())
            # print(h.device, t.device, Gt_mask.device)
            h = torch.cat([h, t], dim=-1)
            # print(h.device, Gt_mask.device)
            h = self.embedding_in(h) # * Gt_mask[:, None] + h_ * (~Gt_mask)[:, None]
        # all_x = [x]
        # all_h = [h]
        for l_idx, layer in enumerate(self.net):
            edge_index = self._connect_edge(x, Gt_mask, batch, bs, n_node)
            edge_type = self._build_edge_type(edge_index, Gt_mask)
            h, x = layer(h, x, edge_index, Gt_mask, edge_attr=edge_type)
            # all_x.append(x)
            # all_h.append(h)
        # x = x.view(bs, n_node, -1)
        # h = h.view(bs, n_node, -1)
        h = self.embedding_out(h)
        h = h * Gt_mask[:, None] + h_ * (~Gt_mask)[:, None]
        # h = F.softmax(h, dim=-1)
        # outputs = {'x': x, 'h': h}
        # if return_all:
        #     outputs.update({'all_x': all_x, 'all_h': all_h})
        # return outputs
        return h, x