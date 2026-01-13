import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from problem_set import device

class AddConstant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def forward(self, x):
        return x + self.value

class DisassemblyModel(nn.Module):

    def __init__(self, input_dim, model_type, embedding_dim, action_dim):
        super().__init__()
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim

        if model_type == "ppo":
            self.state_proj = nn.Sequential(
                nn.Linear(input_dim, input_dim*2), nn.GELU(),
                nn.Linear(input_dim*2, embedding_dim), nn.LayerNorm(embedding_dim)
            )
            self.policy_proj = nn.Linear(embedding_dim, action_dim, bias=False)
            self.value_proj = nn.Linear(embedding_dim, 1)

        elif model_type == "enc_dec":
            self.encoder = EnhancedEncoder(input_dim)
            self.decoder = FixedAttnDecoder(action_dim=action_dim, embedding_dim=embedding_dim)

        elif model_type == "full_gcn_hn":
            self.encoder = GCNEncoder(input_dim, embedding_dim=embedding_dim)
            self.hypernet = DynamicHyperNet(embedding_dim=embedding_dim)
            self.decoder = AdaptiveDecoder(embedding_dim=embedding_dim)

        elif model_type == "full_gat_hn":
            self.encoder = EnhancedEncoder(input_dim)
            self.hypernet = DynamicHyperNet(embedding_dim=embedding_dim)
            self.decoder = AdaptiveDecoder(embedding_dim=embedding_dim)
        
        elif model_type == "wo_multiscale":
            self.encoder = WOEncoder(input_dim)
            self.hypernet = DynamicHyperNet(embedding_dim=embedding_dim)
            self.decoder = AdaptiveDecoder(embedding_dim=embedding_dim)
        
        elif model_type == "full_gat_mlp_hn":
            self.encoder = EnhancedEncoder(input_dim)
            self.hypernet = DynamicHyperNet(embedding_dim=embedding_dim)
            self.decoder = MLPDecoderNoAttn(embedding_dim=embedding_dim)
        
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        self.__init__weights()
        self.to(device)
    
    def __init__weights(self):
        modules = []
        if hasattr(self, "encoder"):
            modules.append(self.encoder)
        if hasattr(self, "hypernet"):
            modules.append(self.hypernet)
        if self.model_type in ["enc_dec", "ppo"]:
            if hasattr(self, "decoder"):
                modules.append(self.decoder)

        for module in modules:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
    
    def forward(self, data, preference):
        if self.model_type == "ppo":
            h_state = self.state_proj(data.x.mean(dim=0))
            policy_logits = self.policy_proj(h_state) 
            policy_logits = policy_logits.masked_fill(~data.mask, -1e9)

            value = self.value_proj(h_state).squeeze(-1)

            return policy_logits, value
        
        elif self.model_type == "enc_dec":
            node_emb, graph_emb = self.encoder(data)
            logits, value = self.decoder(node_emb, graph_emb, data.mask)
            return logits, value
        
        elif self.model_type in ["full_gat_hn", "full_gcn_hn", "wo_multiscale", "full_gat_mlp_hn"]:
            node_emb, graph_emb = self.encoder(data)
            params = self.hypernet(preference)
            logits, value = self.decoder(node_emb, graph_emb, data.mask, params)
            return logits, value
        
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
    


class EnhancedEncoder(nn.Module):

    def __init__(self, input_dim, embedding_dim, gat_heads=4, gat_layers=3, dropout_rate=0.1):

        super().__init__()
        

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim*2),
            nn.LayerNorm(embedding_dim*2),
            nn.GELU(),
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )


        self.gat_layers = nn.ModuleList([
            GATConv(embedding_dim,
                    embedding_dim//gat_heads,
                    heads=gat_heads,
                    dropout=0.1,
                    add_self_loops=False)
            for _ in range(gat_layers)
        ])

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
            for _ in range(gat_layers)
        ])
        

        self.graph_pool = GATConv(embedding_dim, embedding_dim, heads=1, concat=False)

        self.graph_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )

        self.to(device) 
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.to(device)
        edge_index = edge_index.to(device)  
        

        x = self.feature_proj(x)  


        if edge_index.shape[1] == 0:
            graph_emb = x.mean(dim=0)
            graph_emb = self.graph_proj(graph_emb)
            return x, graph_emb
        
        for gat, adapter in zip(self.gat_layers, self.adapters):
            x_residual = x
            x = F.gelu(gat(x, edge_index))
            if x.shape != x_residual.shape:
                x_residual = nn.Linear(x_residual.shape[-1], x.shape[-1])(x_residual)
            x = adapter(x) + x_residual

        if x.shape[0] > 0:
            graph_emb = self.graph_pool(x, edge_index)
            graph_emb = graph_emb.mean(dim=0)
            graph_emb = self.graph_proj(graph_emb)
        else:
            graph_emb = torch.zeros(x.shape[-1], device=x.device)

        return x, graph_emb

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, layers=3, dropout_rate=0.1):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )

        self.convs = nn.ModuleList([GCNConv(embedding_dim, embedding_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = F.gelu(conv(x, edge_index))
            x = norm(x + x_res)
            x = self.dropout(x)
        
        graph_emb = x.mean(dim=0)
        return x, graph_emb


class WOEncoder(nn.Module):

    def __init__(self, input_dim, embedding_dim, gat_heads=4, gat_layers=3, dropout_rate=0.1):

        super().__init__()
        
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.gat_layers = nn.ModuleList([
            GATConv(embedding_dim,
                    embedding_dim//gat_heads,
                    heads=gat_heads,
                    dropout=0.1,
                    add_self_loops=False)
            for _ in range(gat_layers)
        ])

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
            for _ in range(gat_layers)
        ])
        
        self.graph_pool = GATConv(embedding_dim, embedding_dim, heads=1, concat=False)

        self.graph_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )

        self.to(device) 
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.to(device)
        edge_index = edge_index.to(device)  
        
        x = self.feature_proj(x)  

        if edge_index.shape[1] == 0:
            graph_emb = x.mean(dim=0)
            graph_emb = self.graph_proj(graph_emb)
            return x, graph_emb
        
        for gat, adapter in zip(self.gat_layers, self.adapters):
            x_residual = x
            x = F.gelu(gat(x, edge_index))
            if x.shape != x_residual.shape:
                x_residual = nn.Linear(x_residual.shape[-1], x.shape[-1])(x_residual)
            x = adapter(x) + x_residual

        if x.shape[0] > 0:
            graph_emb = self.graph_pool(x, edge_index)
            graph_emb = graph_emb.mean(dim=0)
            graph_emb = self.graph_proj(graph_emb)
        else:
            graph_emb = torch.zeros(x.shape[-1], device=x.device)

        return x, graph_emb

class DynamicHyperNet(nn.Module):
    def __init__(self, embedding_dim=128, n_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        self.pref_encoder = nn.Sequential(
            nn.Linear(3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048)
        )

        self.param_generator = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, self._calculate_total_params())
        )
        

    def _calculate_total_params(self):

        # 注意力参数（Q/K/V/Out）
        attn_params = 4 * (self.embedding_dim * self.embedding_dim)



        #策略头参数
        policy_params = self.embedding_dim * 81

        #价值头参数
        value_params = self.embedding_dim * 1

        return attn_params + policy_params + value_params

        
    def forward(self, preference):
        device = preference.device
        
        if preference.dim() == 1:
            preference = preference.unsqueeze(0)

        B = preference.size(0) 
        


        h = self.pref_encoder(preference)  #(B, 512)  ->  (1, 512)


        #生成所有参数
        all_params = self.param_generator(h)  #(B, total_params)  
        

        #分块解析参数
        ptr = 0
        params_dict = {}

        attn_size = 4 * self.embedding_dim**2
        attn_params = all_params[:, ptr:ptr+attn_size]
        ptr += attn_size

        #分解为Q/K/V/Out矩阵，无偏置
        q_weight = attn_params[:, :self.embedding_dim**2].view(B, self.embedding_dim, self.embedding_dim)
        k_weight = attn_params[:, self.embedding_dim**2:2*self.embedding_dim**2].view(B, self.embedding_dim, self.embedding_dim)
        v_weight = attn_params[:, 2*self.embedding_dim**2:3*self.embedding_dim**2].view(B, self.embedding_dim, self.embedding_dim)
        out_weight = attn_params[:, 3*self.embedding_dim**2:].view(B, self.embedding_dim, self.embedding_dim)

        params_dict.update({
            'q_weight': q_weight,
            'k_weight' : k_weight,
            'v_weight' : v_weight,
            'out_weight' : out_weight
        })


        #策略头参数
        policy_weight = all_params[:, ptr:ptr+self.embedding_dim*81].view(B, self.embedding_dim, 81)
        ptr += self.embedding_dim*81
        params_dict['policy_weight'] = policy_weight

        #价值头参数
        value_weight = all_params[:, ptr:ptr+self.embedding_dim].view(B, self.embedding_dim, 1)
        params_dict['value_weight'] = value_weight


        for key in params_dict:
            params_dict[key] = params_dict[key].to(device)

        return params_dict



class AdaptiveDecoder(nn.Module):
    def __init__(self, embedding_dim=128, n_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads


    def forward(self, node_emb, graph_emb, mask, hyper_params):

        device = node_emb.device

        if node_emb.dim() == 2:
            node_emb = node_emb.unsqueeze(0)  #[1,23,128]
        
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)  #[1,128]
        
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)  #[1,23]
        
        B, N, D = node_emb.shape  #[1,23,128]

        graph_emb = graph_emb.unsqueeze(1)  #[B, 1, D]

        for key in hyper_params:
            if isinstance(hyper_params[key], torch.Tensor):
                hyper_params[key] = hyper_params[key].to(device)    

      
        q = torch.einsum('bnd,bdd->bnd', graph_emb, hyper_params['q_weight'])
        q = q.view(B, 1, self.n_heads, self.head_dim)
        k = torch.einsum('bnd,bdd->bnd', node_emb, hyper_params['k_weight'])
        k = k.view(B, N, self.n_heads, self.head_dim)
        v = torch.einsum('bnd,bdd->bnd', node_emb, hyper_params['v_weight'])
        v = v.view(B, N, self.n_heads, self.head_dim)

        attn = torch.einsum('bnhd, bmhd->bnmh', q, k) / (self.head_dim**0.5)  #[1,1,23,4]

        if mask is not None:  #[1, 23]
            #[1, 1, 23, 1]
            attention_mask = mask.unsqueeze(1).unsqueeze(-1)
            attn = attn.masked_fill(~attention_mask, -1e9)  #[1,1,23,1]
        
        attn = F.softmax(attn, dim=2)
        context = torch.einsum('bnmh, bmhd->bnhd', attn, v)  #[1,1,4,32]
        context = context.reshape(B,1,D) #[1,1,128]

        out = torch.einsum('bnd,bdd->bnd', context, hyper_params['out_weight'])  #[1,1,128]

        h = out.squeeze(1)  # [B, D]

        policy_logits = torch.einsum('bd, bdk->bk', h, hyper_params['policy_weight'])
        policy_logits = policy_logits.masked_fill(~mask.squeeze(1), -1e9)

        value = torch.einsum('bd,bd->b', h, hyper_params['value_weight'].squeeze(-1))

        return policy_logits, value


class FixedAttnDecoder(nn.Module):
    def __init__(self, action_dim, embedding_dim=128, n_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.o_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.policy_proj = nn.Linear(embedding_dim, action_dim, bias=False)
        self.value_proj = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, node_emb, graph_emb, mask):
        if node_emb.dim() == 2:  #[16, 128]
            node_emb = node_emb.unsqueeze(0)  #[1,16,128]
        if graph_emb.dim() == 1:  #[128]
            graph_emb = graph_emb.unsqueeze(0)  #[1,128]
        if mask.dim() == 1:
            mask = mask.unsqueeze(0) #[1,16]
        
        B, N, D = node_emb.shape
        g = graph_emb.unsqueeze(1) #[1, 1, 128]

        # Q/K/V
        Q = self.q_proj(g).view(B, 1, self.n_heads, self.head_dim)
        K = self.k_proj(node_emb).view(B, N, self.n_heads, self.head_dim)
        V = self.v_proj(node_emb).view(B, N, self.n_heads, self.head_dim)

        attn = torch.einsum('bqhd, bnhd->bqnh', Q, K) / (self.head_dim**0.5)

        if mask is not None:
            attention_mask = mask.unsqueeze(1).unsqueeze(-1)
            attn = attn.masked_fill(~attention_mask, -1e9)

        attn = F.softmax(attn, dim=2)
        context = torch.einsum('bqnh, bnhd->bqhd', attn, V)
        context = context.reshape(B,1,D)

        out = self.o_proj(context)  #[1, 1, 128]

        h = out.squeeze(1)  #[1, 128]

        policy_logits = self.policy_proj(h)
        policy_logits = policy_logits.masked_fill(~mask.squeeze(1), -1e9)

        value = self.value_proj(h).squeeze(-1)

        return policy_logits, value

class MLPDecoderNoAttn(nn.Module):
    def __init__(self, embedding_dim, action_dim, hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def forward(self, node_emb, graph_emb, mask, hyper_params):
        if node_emb.dim() == 2:      # [N, D]
            node_emb = node_emb.unsqueeze(0)   # [1, N, D]
        if graph_emb.dim() == 1:     # [D]
            graph_emb = graph_emb.unsqueeze(0) # [1, D]
        if mask is not None and mask.dim() == 1:  # [N] 或 [1]
            mask = mask.unsqueeze(0)              # [1, N] 或 [1, 1]
        
        B, N, D = node_emb.shape
        device = node_emb.device

        for k, v in hyper_params.items():
            if isinstance(v, torch.Tensor):
                hyper_params[k] = v.to(device)

        W_mlp = hyper_params["mlp_weight"]      # [B, 2D, D]
        W_p   = hyper_params["policy_weight"]   # [B, D, A]
        W_v   = hyper_params["value_weight"]    # [B, D, 1]


        if mask is not None and mask.shape[1] == N:
            mask_float = mask.float()                        # [B, N]
            node_sum = (node_emb * mask_float.unsqueeze(-1)).sum(dim=1)  # [B, D]
            denom = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
            node_global = node_sum / denom                  # [B, D]
        else:
            node_global = node_emb.mean(dim=1)              # [B, D]

        x = torch.cat([graph_emb, node_global], dim=-1)      # [B, 2D]

        h = torch.einsum("bd,bdk->bk", x, W_mlp)             # [B, D]
        h = F.gelu(h)

        policy_logits = torch.einsum("bd,bdk->bk", h, W_p)   # [B, A]
        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask.squeeze(1), -1e9)

        value_weight = W_v.squeeze(-1)                       # [B, D]
        value = torch.einsum("bd,bd->b", h, value_weight)    # [B]

        return policy_logits, value