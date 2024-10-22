import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=256,
            dropout=0.1
    ):
        super().__init__()

        """
        Here you should write simple 2-layer MLP consisting:
        2 Linear layers, GELU activation, Dropout and LayerNorm. 
        Do not forget to send a skip-connection right after projection and before LayerNorm.
        The whole structure should be in the following order:
        [Linear, GELU, Linear, Dropout, Skip, LayerNorm]
        """
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, projection_dim),
            nn.Dropout(p=dropout)
        )
        self.skip = nn.Linear(embedding_dim, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        """
        Perform forward pass, do not forget about skip-connections.
        """
        return self.norm(self.projection(x) + self.skip(x))
