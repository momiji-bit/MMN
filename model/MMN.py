from typing import Type, Tuple, Optional, Set, List, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, Mlp, DropPath


"""Motion-guided Skeletal-Temporal Former"""

class GraphConvGuidance(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        """
        Graph Convolution with Learnable Adjacency Matrix.

        Args:
            in_channels (int): Number of input channels (channels of diff_feature, C//2).
            out_channels (int): Number of output channels ((C//4)*2).
            num_nodes (int): Number of nodes (V).
        """
        super(GraphConvGuidance, self).__init__()

        # Learnable adjacency matrix, initialized as identity
        self.adj = nn.Parameter(torch.eye(num_nodes))
        # Linear transformation
        self.fc = nn.Linear(in_channels, out_channels)
        # Activation function
        self.tanh = nn.Tanh()
        # BatchNorm2d expects input of shape [B, out_channels, T, V]
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Permute to [B, T, V, C] for graph convolution along node dimension
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T, V, C]
        # Graph convolution using learnable adjacency matrix
        x = torch.einsum('vw,btwc->btwc', self.adj, x)  # [B, T, V, C]
        # Linear transformation + activation
        x = self.fc(x)  # [B, T, V, out_channels]
        x = self.tanh(x)
        # Rearrange to [B, out_channels, T, V] for BatchNorm2d
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.bn(x)

        return x


class MGSTFormer(nn.Module):
    def __init__(self, in_channels=2, num_points=44, kernel_size=3, num_heads=4,
                 drop=0., drop_path=0., mlp_ratio=2.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Multi-branch Motion Network (MMN) Block.

        Args:
            in_channels (int): Number of input channels.
            num_points (int): Number of nodes (default: 50).
            kernel_size (int): Temporal convolution kernel size (default: 7).
            num_heads (int): Number of attention heads (default: 32).
            drop (float): Dropout rate for projection.
            drop_path (float): Stochastic depth rate (default: 0.).
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            act_layer (nn.Module): Activation function class (default: GELU).
            norm_layer (nn.Module): Normalization layer class (default: LayerNorm).
        """
        super(MGSTFormer, self).__init__()

        # -------------------- Linear & Normalization --------------------
        self.mapping = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.norm_1 = norm_layer(in_channels)

        # -------------------- Skeleton Branch --------------------
        # Learnable adjacency tensor: [num_heads, num_points, num_points]
        self.gconv = nn.Parameter(torch.zeros(num_heads, num_points, num_points))
        trunc_normal_(self.gconv, std=.02)

        # -------------------- Temporal Branch --------------------
        # Output channels: in_channels // 4
        self.tconv = nn.Conv2d(in_channels // 4, in_channels // 4,
                               kernel_size=(kernel_size, 1),
                               padding=((kernel_size - 1) // 2, 0),
                               groups=num_heads)

        # -------------------- Motion Branch --------------------
        self.Motion = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2,
                      kernel_size=(3, 1), stride=1,
                      padding=(1, 0), groups=in_channels // 2, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU()
        )

        # -------------------- Projection & Residual --------------------
        self.proj = nn.Linear(in_channels, in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(in_features=in_channels,
                       hidden_features=int(mlp_ratio * in_channels),
                       act_layer=act_layer, drop=drop)

        # -------------------- Guidance Modules --------------------
        # Use diff branch to generate gamma & beta for both TConv and GConv.
        # Gamma scales, Beta shifts. Tanh ensures bounded mapping.
        self.MTM = nn.Sequential(
            nn.Conv2d(in_channels // 2, (in_channels // 4) * 2,
                      kernel_size=(3, 1), padding=(1, 0)),
            nn.Tanh(),
            nn.BatchNorm2d((in_channels // 4) * 2)
        )

        self.MSM = GraphConvGuidance(in_channels // 2, (in_channels // 4) * 2, num_points)

        # -------------------- Adaptive Fusion Module --------------------
        # Concatenate outputs: [TConv_out, GConv_out, Diff_feature]
        # Channels: C//4 + C//4 + C//2 = C
        self.aggregation_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 3, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Final fusion projection: ensure output channels = in_channels
        self.aggregation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

    def interpolate_diff(self, x, order=0, dim=2):
        """
        Compute temporal differences with padding.

        Args:
            x (Tensor): Input tensor.
            order (int): Number of differential orders.
            dim (int): Dimension along which to compute differences (default: 2 for T).

        Returns:
            Tensor: Differential-augmented tensor.
        """
        out = x
        for _ in range(order):
            out = torch.diff(out, dim=dim)
            # For 4D tensor, F.pad args: (pad_left, pad_right, pad_top, pad_bottom)
            out = F.pad(out, (0, 0, 0, 1), mode='replicate')
        return out

    def forward(self, X_feat):
        """
        Forward pass of MMNBlock.

        Args:
            X_feat (Tensor): Input tensor of shape [B, C, T, V].

        Returns:
            Tensor: Output tensor of shape [B, C, T, V].
        """
        B, C, T, V = X_feat.shape

        # -------------------- Linear & LN & Res  --------------------
        X_feat = X_feat.permute(0, 2, 3, 1).contiguous()  # [B, T, V, C]
        X_feat_res = X_feat

        X_in = self.mapping(self.norm_1(X_feat)).permute(0, 3, 1, 2).contiguous()  # [B, C, T, V]
        f_st, X_in_motion = torch.split(X_in, [C // 2, C // 2], dim=1)
        X_in_tc = torch.chunk(f_st, 2, dim=1)  # [B, C//4, T, V]

        # -------------------- Skeleton Branch --------------------
        X_gc_ = []
        X_in_gc = torch.chunk(X_in_tc[0], self.gconv.shape[0], dim=1)
        for i in range(self.gconv.shape[0]):
            z = torch.einsum('n c t u, v u -> n c t v', X_in_gc[i], self.gconv[i])
            X_gc_.append(z)
        X_gc = torch.cat(X_gc_, dim=1)  # [B, C//4, T, V]

        # -------------------- Temporal Branch --------------------
        X_tc = self.tconv(X_in_tc[1])  # [B, C//4, T, V]

        # -------------------- Motion Branch --------------------
        X_delta = self.interpolate_diff(X_in_motion, order=1)
        motion = self.Motion(X_delta)  # [B, C//2, T, V]

        # ------------- Motion-guided Skeletal Modulation (MSM) ------------------
        Z_s = self.MSM(motion)
        gamma_s, beta_s = torch.chunk(Z_s, 2, dim=1)

        mean_g = X_gc.mean(dim=(2, 3), keepdim=True)
        std_g = X_gc.std(dim=(2, 3), keepdim=True)
        X_gcm = (X_gc - mean_g) / (std_g + 1e-6) * (1 + gamma_s) + beta_s

        # ------------- Motion-guided Temporal Modulation (MTM) ------------------
        Z_t = self.MTM(motion)
        gamma_t, beta_t = torch.chunk(Z_t, 2, dim=1)

        mean_t = X_tc.mean(dim=(2, 3), keepdim=True)
        std_t = X_tc.std(dim=(2, 3), keepdim=True)
        X_tcm = (X_tc - mean_t) / (std_t + 1e-6) * (1 + gamma_t) + beta_t

        # -------------------- Aggregation --------------------
        X_agg = torch.cat([X_tcm, X_gcm, motion], dim=1)  # [B, C, T, V]
        agg_weights = self.aggregation_gate(X_agg)  # [B, 3, T, V]
        weighted_t = X_tc * agg_weights[:, 0:1, :, :]
        weighted_g = X_gc * agg_weights[:, 1:2, :, :]
        weighted_d = motion * agg_weights[:, 2:3, :, :]
        agg_feature = torch.cat([weighted_t, weighted_g, weighted_d], dim=1)
        agg_feature = self.aggregation(agg_feature)

        # -------------------- FFN --------------------
        X_f = self.proj_drop(self.proj(agg_feature.permute(0, 2, 3, 1).contiguous()))
        X_f = X_feat_res + self.drop_path(X_f)

        X_f = X_f + self.drop_path(self.mlp(self.norm_2(X_f)))
        X_f = X_f.permute(0, 3, 1, 2).contiguous()

        return X_f


"""Motion-guided Feature Modulation"""

class MFM_Layer(nn.Module):
    def __init__(
            self, depth, in_channels, out_channels, num_points=44, kernel_size=3, num_heads=4,
            drop=0., drop_path=0., mlp_ratio=2.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MFM_Layer, self).__init__()
        blocks = []
        for index in range(depth):
            blocks.append(
                MGSTFormer(
                    in_channels=in_channels if index == 0 else out_channels,
                    num_points=num_points,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    drop=drop,
                    drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        output = input
        for block in self.blocks:
            output = block(output)
        return output


"""Motion Consistency Learning"""

class MultiScaleSpatioTemporalFeatureConstructor(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=2, dilation=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv2d(dim_in, dim_out, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1),
                                   dilation=(dilation, 1), padding_mode='replicate')
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x = self.bn(self.reduction(x))
        return x

class MotionConsistencyLearning(nn.Module):
    def __init__(self, in_channels, dropout=0.1):
        """
        Args:
            in_channels (int): Number of input feature channels.
            dropout (float): Dropout probability, used to reduce overfitting.
        """
        super(MotionConsistencyLearning, self).__init__()

        # Downsample features at multiple scales with BN and ReLU
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(8, 1), stride=(8, 1)),
            nn.BatchNorm2d(in_channels),  # Batch Normalization
            nn.ReLU(inplace=True)  # ReLU activation
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(4, 1), stride=(4, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # f4 is already at the target resolution, but still processed with BN and ReLU
        self.f4_proc = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Deep Gating Network: with intermediate layers to increase capacity
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, kernel_size=1)
        )

        self.dropout = nn.Dropout(dropout)  # Dropout layer to mitigate overfitting

        # Residual connection branch for stable training
        self.res_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (tuple):
                f1: [B, C, T, V]
                f2: [B, C, T//2, V]
                f3: [B, C, T//4, V]
                f4: [B, C, T//8, V]

        Returns:
            torch.Tensor: [B, C, T//8, V] — fused feature representation
        """
        f1, f2, f3, f4 = x

        # Downsample and process features at different scales
        f1_ds = self.down1(f1)  # [B, C, T//8, V]
        f2_ds = self.down2(f2)  # [B, C, T//8, V]
        f3_ds = self.down3(f3)  # [B, C, T//8, V]
        f4_proc = self.f4_proc(f4)  # Process f4 with BN and ReLU (no downsampling)

        # Concatenate all scale features along the channel dimension
        fused_features = torch.cat([f1_ds, f2_ds, f3_ds, f4_proc], dim=1)  # [B, 4C, T//8, V]

        # Compute adaptive gating weights: [B, 4, T//8, V]
        gate_weights = self.gate_conv(fused_features)
        gate_weights = F.softmax(gate_weights, dim=1)  # Normalize along scale dimension

        # Weighted fusion using gating weights
        fused_output = (gate_weights[:, 0:1, :, :] * f1_ds +
                        gate_weights[:, 1:2, :, :] * f2_ds +
                        gate_weights[:, 2:3, :, :] * f3_ds +
                        gate_weights[:, 3:4, :, :] * f4_proc)

        fused_output = self.dropout(fused_output)  # Apply Dropout to prevent overfitting

        # Add residual connection to enhance gradient flow and model stability
        identity = self.res_conv(f4_proc)
        output = fused_output + identity

        return output


"""Motion-guidedModulation Network (MMN)"""

class MMN(nn.Module):
    def __init__(self, in_channels=2, depths=(3, 3, 3, 3), channels=(96, 96, 96, 96), num_classes=52,
                 embed_dim=96, num_people=1, num_frames=64, num_points=44, kernel_size=3, num_heads=4,
                 head_drop=0., drop=0., drop_path=0., mlp_ratio=2.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, index_t=True, global_pool='avg',):

        super(MMN, self).__init__()

        assert len(depths) == len(channels), "For each stage a channel dimension must be given."
        assert global_pool in ["avg", "max"], f"Only avg and max is supported but {global_pool} is given"
        self.num_classes: int = num_classes
        self.head_drop = head_drop
        self.index_t = index_t
        self.embed_dim = embed_dim

        if self.head_drop != 0:
            self.dropout = nn.Dropout(p=self.head_drop)
        else:
            self.dropout = None

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=1, stride=1, padding=0),
            act_layer(),
            nn.Conv2d(in_channels=2 * in_channels, out_channels=3 * in_channels, kernel_size=1, stride=1, padding=0),
            act_layer(),
            nn.Conv2d(in_channels=3 * in_channels, out_channels=embed_dim, kernel_size=1, stride=1, padding=0)
        )

        if self.index_t:
            self.STPE = nn.Parameter(torch.zeros(embed_dim, num_points * num_people))
            trunc_normal_(self.STPE, std=.02)
        else:
            self.STPE_ = nn.Parameter(
                torch.zeros(1, embed_dim, num_frames, num_points * num_people))
            trunc_normal_(self.STPE_, std=.02)

        # Init blocks
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        MFM = []
        DS = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            MFM.append(
                MFM_Layer(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    num_points=num_points * num_people,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    drop=drop,
                    drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                )
            )
            if index != len(depths) - 1:
                DS.append(
                    MultiScaleSpatioTemporalFeatureConstructor(channels[index], channels[index + 1], kernel_size=kernel_size)
                )
        self.MFM = nn.ModuleList(MFM)
        self.DSs = nn.ModuleList(DS)

        self.global_pool: str = global_pool

        self.head = nn.Linear(channels[-1], num_classes)

        self.fusion = MotionConsistencyLearning(in_channels=channels[-1])

    def forward_features(self, X_feat):
        X_f_L = []
        DSs = [X_feat]

        ds_output = X_feat
        for ds in self.DSs:
            ds_output = ds(ds_output)
            DSs.append(ds_output)

        for X_f, mfm_layer in zip(DSs, self.MFM):
            X_f_L.append(mfm_layer(X_f))

        X_z = self.fusion(X_f_L)
        return X_z

    def classifier(self, input, pre_logits=False):
        if self.global_pool == "avg":
            input = input.mean(dim=(2, 3))
        elif self.global_pool == "max":
            input = torch.amax(input, dim=(2, 3))
        if self.dropout is not None:
            input = self.dropout(input)
        return input if pre_logits else self.head(input)

    def feature_embedding(self, X_raw, index_t):
        B, C, T, V, M = X_raw.shape

        X_raw = X_raw.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)  # [B, C, T, M * V]

        output = self.projection(X_raw)

        if self.index_t:
            te = torch.zeros(B, T, self.embed_dim).to(output.device)  # B, T, C
            div_term = torch.exp(
                (torch.arange(0, self.embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / self.embed_dim))).to(
                output.device)
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            X_feat = output + torch.einsum('b t c, c v -> b c t v', te, self.STPE)
        else:
            X_feat = output + self.STPE_

        return X_feat

    def forward(self, input, index_t):
        X_feat = self.feature_embedding(input, index_t)
        X_z = self.forward_features(X_feat)
        return self.classifier(X_z)


def MMN_(**kwargs):
    return MMN(
        depths=(3, 3, 3, 3),
        channels=(96, 96, 96, 96),
        embed_dim=96,
        **kwargs
    )

