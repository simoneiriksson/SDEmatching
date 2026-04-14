import torch
import torch.nn as nn

class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dim, time_embed_dim, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, model_dim)
        )
        self.data_proj = nn.Linear(input_dim, model_dim)

    def forward(self, t, x):
        #print(f"{t.shape = }")
        t_embed = self.time_mlp(t[:,:,None])[:,:,0,:]          # (B, T, model_dim)
        x_embed = self.data_proj(x)         # (B, T, model_dim)
        #print(f"{t_embed.shape = }")
        #print(f"{x_embed.shape = }")
        
        return t_embed + x_embed            # (B, T, model_dim)

class TransformerContextEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, n_heads, n_layers, time_embed_dim):
        super().__init__()
        self.embedding = TimeSeriesEmbedding(input_dim, time_embed_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=4 * model_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, t, x):
        # t: (B, T, 1), x: (B, T, obs_dim)
        z = self.embedding(t, x)           # (B, T, model_dim)
        context = self.encoder(z)          # (B, T, model_dim)
        return context                     # contextualized embeddings

class TimeDecoder(nn.Module):
    def __init__(self, model_dim, state_dim, time_bias_scale=0.1):
        super().__init__()
        self.query_embed = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
        self.output = nn.Linear(model_dim, 2 * state_dim)
        self.scale = time_bias_scale

    def forward(self, context, t_star, t_context):
        """
        context:    (B, T, model_dim)
        t_star:     (B, 1)  — query time
        t_context:  (B, T, 1) — observed times
        """
        # Embed the query time
        q = self.query_embed(t_star.unsqueeze(1)).unsqueeze(1)  # (B, 1, model_dim)

        # Dot product attention
        scores = torch.einsum("bij,bkj->bik", q, context)       # (B, 1, T)

        # Add relative time bias
        #print(f"{t_star.shape = }")
        #print(f"{t_context.shape = }")
        delta_t = t_star[:,None, None] - t_context               # (B, T, 1)
        time_bias = -torch.abs(delta_t.squeeze(-1)) / self.scale  # (B, T)
        scores += time_bias.unsqueeze(1)                        # (B, 1, T)

        # Softmax + attention
        weights = torch.softmax(scores, dim=-1)                 # (B, 1, T)
        pooled = torch.einsum("bik,bkj->bij", weights, context) # (B, 1, model_dim)
        pooled = pooled.squeeze(1)

        # Output to mean and log-variance
        out = self.output(pooled)
        mu, log_sigma = out.chunk(2, dim=-1)
        return mu, log_sigma

class TransformerLatentModel(nn.Module):
    def __init__(self, obs_dim, state_dim, model_dim=128, time_embed_dim=32, n_heads=4, n_layers=8):
        super().__init__()
        self.encoder = TransformerContextEncoder(obs_dim, model_dim, n_heads, n_layers, time_embed_dim)
        self.decoder = TimeDecoder(model_dim, state_dim)

    def forward(self, t_star, data):
        t_obs = data[:, :, 0:1]    # shape: (B, T, 1)
        x_obs = data[:, :, 1:]     # shape: (B, T, obs_dim)
        context = self.encoder(t_obs, x_obs)                       # (B, T, model_dim)
        mu, log_sigma = self.decoder(context, t_star, t_obs)       # <== now passes t_obs too
        return torch.cat([mu, log_sigma], dim=1)
