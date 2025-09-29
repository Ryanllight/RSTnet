
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

# --- Base classes from Swin-Unet ---

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

# --- RSTnet specific modules ---

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class ConvEncoder(nn.Module):
    def __init__(self, in_chans=3, dims=[96, 192, 384, 768]):
        super().__init__()
        self.layers = nn.ModuleList()
        current_in = in_chans
        for i, dim in enumerate(dims):
            # The paper mentions 3x3 conv layers to extract local detail features.
            # For the first layer, it's typically a stride 2 or 4 to reduce resolution.
            # Assuming a similar downsampling strategy as Swin-Unet's patch embedding for initial stages.
            # The paper states 'three consecutive 3x3 convolutional layers to extract local detail features' for Conv-Encoder.
            # This implies a sequence of conv layers, not necessarily downsampling at each step.
            # However, to match the feature map resolutions for skip connections, downsampling is needed.
            # Let's align the downsampling with Swin Transformer stages.
            stride = 2 if i > 0 else 4 # Initial downsampling for the first stage to match Swin-Unet's first stage output resolution
            self.layers.append(nn.Sequential(
                nn.Conv2d(current_in, dim, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ))
            current_in = dim

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

class RCASC(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(RCASC, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x)
        return x * se_weight + x # Residual connection

class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# --- Main RSTnet Model ---

class RSTnet(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=9,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) # Corresponds to the deepest Swin feature
        self.mlp_ratio = mlp_ratio

        # Swin Transformer Encoder
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.swin_encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.swin_encoder_layers.append(layer)

        # CBAM modules for Swin Encoder (applied after each Swin block output)
        self.cbam_modules = nn.ModuleList([
            CBAM(in_planes=int(embed_dim * 2 ** i_layer)) for i_layer in range(self.num_layers)
        ])

        # Conv Encoder
        conv_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8] # Match Swin-Unet's embedding dimensions
        self.conv_encoder = ConvEncoder(in_chans=in_chans, dims=conv_dims)

        # Conv Bottleneck (applied to the deepest Swin feature)
        self.conv_bottleneck = ConvBottleneck(in_channels=self.num_features, out_channels=self.num_features)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.rcasc_modules = nn.ModuleList()
        # The concat_back_dim layers are used to adjust channels after concatenation in the original Swin-Unet.
        # In RSTnet, we'll use 1x1 convs for fusion after RCASC and before passing to the decoder layer.
        self.fusion_convs = nn.ModuleList()

        for i_layer in range(self.num_layers):
            # RCASC for skip connections (applied to Swin features before fusion)
            # The RCASC modules are for each skip connection from Swin encoder to decoder.
            # There are num_layers - 1 skip connections (excluding the bottleneck).
            if i_layer < self.num_layers - 1:
                self.rcasc_modules.append(RCASC(in_channels=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))))

            # Fusion convs for combining upsampled feature, RCASC-enhanced Swin skip, and Conv skip
            # The input to the fusion conv will be (upsampled_dim + rcasc_swin_dim + conv_skip_dim)
            # The output should be the dim for the current decoder stage.
            current_dim_decoder = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            if i_layer > 0: # For all but the first decoder stage (which is just upsampling from bottleneck)
                # The input to fusion_conv will be (current_dim_decoder + current_dim_decoder) if we assume
                # conv_skip and swin_skip are both 'current_dim_decoder' after RCASC and upsampling.
                # However, the paper states 

                # The paper states that the output of RCASC is added to the upsampled feature.
                # And the Conv-Encoder feature is also added. So the fusion is additive.
                # We need a conv layer to adjust the channels of the conv_skip to match the swin_skip.
                # Let's assume the conv_skip dimensions are already aligned or can be adjusted implicitly.
                # The original Swin-Unet concat_back_dim was for concatenating swin_skip and upsampled_x.
                # Here, we have three components: upsampled_x, rcasc_swin_skip, conv_skip.
                # The paper implies an additive fusion. So, the fusion_convs should handle the combined channels if needed.
                # For now, let's assume direct addition is possible if dimensions match, or use a 1x1 conv to align channels.
                # The current_dim_decoder is the target dimension for the decoder stage.
                # The conv_skip from ConvEncoder will have `conv_dims[self.num_layers - 1 - i]` channels.
                # The swin_skip (after RCASC) will have `current_dim_decoder` channels.
                # The upsampled_x will also have `current_dim_decoder` channels.
                # So, we need to ensure conv_skip matches `current_dim_decoder` before addition.
                self.fusion_convs.append(nn.Conv2d(conv_dims[self.num_layers - 1 - i], current_dim_decoder, kernel_size=1))
            else:
                self.fusion_convs.append(nn.Identity()) # No fusion conv for the first decoder stage

            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                           self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths_decoder[self.num_layers - 1 - i_layer],
                                         num_heads=num_heads[self.num_layers - 1 - i_layer],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.decoder_layers.append(layer_up)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim) # Final normalization before final upsample
        self.final_upsample = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                                  dim_scale=4, dim=embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Dual Encoders
        conv_features = self.conv_encoder(x) # List of 2D features
        swin_x, swin_downsample_features = self.forward_swin_encoder(x) # swin_x is bottleneck, swin_downsample_features are list of 1D features

        # Bottleneck
        bottleneck_output = self.conv_bottleneck(swin_x)

        # Decoder
        decoder_output = self.forward_decoder(bottleneck_output, swin_downsample_features, conv_features)

        # Final Output
        out = self.final_upsample(decoder_output)
        # The output of FinalPatchExpand_X4 is (B, L, C_out), need to reshape to (B, C_out, H, W)
        B, L, C_out = out.shape
        H_out = self.patches_resolution[0] * 4 # Assuming 4x upsampling from embed_dim stage
        W_out = self.patches_resolution[1] * 4
        out = out.view(B, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()
        out = self.output(out)
        return out

    def forward_swin_encoder(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        downsample_features = []
        for i, layer in enumerate(self.swin_encoder_layers):
            # Store feature before downsampling (skip connection)
            downsample_features.append(x)
            x = layer(x)
            
            # Apply CBAM after each Swin block output (except the last one which is bottleneck)
            if i < self.num_layers - 1: # CBAM is applied to the output of each Swin stage before downsampling
                B, L, C = x.shape
                H_res = self.patches_resolution[0] // (2 ** (i + 1))
                W_res = self.patches_resolution[1] // (2 ** (i + 1))
                x_2d = x.view(B, H_res, W_res, C).permute(0, 3, 1, 2).contiguous()
                x_2d = self.cbam_modules[i](x_2d) # Apply CBAM to 2D feature map
                x = x_2d.flatten(2).transpose(1, 2).contiguous() # Convert back to 1D for next Swin block

        x = self.norm(x) # Normalize bottleneck output
        B, L, C = x.shape
        H_bottleneck = self.patches_resolution[0] // (2 ** (self.num_layers - 1))
        W_bottleneck = self.patches_resolution[1] // (2 ** (self.num_layers - 1))
        x = x.view(B, H_bottleneck, W_bottleneck, C).permute(0, 3, 1, 2).contiguous() # Convert to 2D for ConvBottleneck
        return x, downsample_features

    def forward_decoder(self, x, swin_skips, conv_skips):
        # x is the bottleneck output (B, C, H, W) from ConvBottleneck
        # swin_skips are list of 1D features (B, L, C) from Swin Encoder
        # conv_skips are list of 2D features (B, C, H, W) from Conv Encoder

        # The first decoder layer upsamples the bottleneck output
        x = x.flatten(2).transpose(1, 2).contiguous() # Convert bottleneck output to 1D (B, L, C)

        for i, layer_up in enumerate(self.decoder_layers):
            if i > 0: # For subsequent decoder stages, perform fusion
                # Get skip features from Swin and Conv encoders
                # Swin skip features are in reverse order (deepest to shallowest)
                swin_skip_1d = swin_skips[self.num_layers - 1 - i] # (B, L, C)
                conv_skip_2d = conv_skips[self.num_layers - 1 - i] # (B, C, H, W)

                # Convert swin_skip to 2D for RCASC and fusion
                B, L_swin, C_swin = swin_skip_1d.shape
                H_swin = self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i))
                W_swin = self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i))
                swin_skip_2d = swin_skip_1d.view(B, H_swin, W_swin, C_swin).permute(0, 3, 1, 2).contiguous()

                # Apply RCASC to Swin skip
                rcasc_swin_skip = self.rcasc_modules[i-1](swin_skip_2d) # RCASC is applied to 2D feature

                # Adjust conv_skip channels if necessary and add to rcasc_swin_skip
                fused_skip = rcasc_swin_skip + self.fusion_convs[i](conv_skip_2d)

                # Upsample x to match skip connection resolution and convert to 2D
                B_x, L_x, C_x = x.shape
                H_x = self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i))
                W_x = self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i))
                x_2d = x.view(B_x, H_x, W_x, C_x).permute(0, 3, 1, 2).contiguous()

                # Add fused skip to upsampled x
                x_2d = x_2d + fused_skip
                x = x_2d.flatten(2).transpose(1, 2).contiguous() # Convert back to 1D for next decoder block

            x = layer_up(x)

        x = self.norm_up(x)
        return x

# Wrapper for compatibility with original training script
class RST_SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=9, zero_head=False, vis=False):
        super(RST_SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.rst_net = RSTnet(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                depths_decoder=config.MODEL.SWIN.DEPTHS_DECODER, # Added depths_decoder
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.rst_net(x)
        return logits

    def load_from(self, config):
        # Weight loading logic can be adapted here if needed
        print("Weight loading from pretrained Swin-T not implemented for RSTnet in this script.")
        pass


ily downsampling at each step.
            # However, to match the feature map resolutions for skip connections, downsampling is needed.
            # Let's align the downsampling with Swin Transformer stages.
            stride = 2 if i > 0 else 4 # Initial downsampling for the first stage to match Swin-Unet's first stage output resolution
            self.layers.append(nn.Sequential(
                nn.Conv2d(current_in, dim, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ))
            current_in = dim

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

class RCASC(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(RCASC, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x)
        return x * se_weight + x # Residual connection

class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# --- Main RSTnet Model ---

class RSTnet(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=9,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) # Corresponds to the deepest Swin feature
        self.mlp_ratio = mlp_ratio

        # Swin Transformer Encoder
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.swin_encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.swin_encoder_layers.append(layer)

        # CBAM modules for Swin Encoder (applied after each Swin block output)
        self.cbam_modules = nn.ModuleList([
            CBAM(in_planes=int(embed_dim * 2 ** i_layer)) for i_layer in range(self.num_layers)
        ])

        # Conv Encoder
        conv_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8] # Match Swin-Unet's embedding dimensions
        self.conv_encoder = ConvEncoder(in_chans=in_chans, dims=conv_dims)

        # Conv Bottleneck (applied to the deepest Swin feature)
        self.conv_bottleneck = ConvBottleneck(in_channels=self.num_features, out_channels=self.num_features)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.rcasc_modules = nn.ModuleList()
        # The concat_back_dim layers are used to adjust channels after concatenation in the original Swin-Unet.
        # In RSTnet, we'll use 1x1 convs for fusion after RCASC and before passing to the decoder layer.
        self.fusion_convs = nn.ModuleList()

        for i_layer in range(self.num_layers):
            # RCASC for skip connections (applied to Swin features before fusion)
            # The RCASC modules are for each skip connection from Swin encoder to decoder.
            # There are num_layers - 1 skip connections (excluding the bottleneck).
            if i_layer < self.num_layers - 1:
                self.rcasc_modules.append(RCASC(in_channels=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))))

            # Fusion convs for combining upsampled feature, RCASC-enhanced Swin skip, and Conv skip
            # The input to the fusion conv will be (upsampled_dim + rcasc_swin_dim + conv_skip_dim)
            # The output should be the dim for the current decoder stage.
            current_dim_decoder = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            if i_layer > 0: # For all but the first decoder stage (which is just upsampling from bottleneck)
                # The paper states that the output of RCASC is added to the upsampled feature.
                # And the Conv-Encoder feature is also added. So the fusion is additive.
                # We need a conv layer to adjust the channels of the conv_skip to match the swin_skip.
                # Let's assume the conv_skip dimensions are already aligned or can be adjusted implicitly.
                # The original Swin-Unet concat_back_dim was for concatenating swin_skip and upsampled_x.
                # Here, we have three components: upsampled_x, rcasc_swin_skip, conv_skip.
                # The paper implies an additive fusion. So, the fusion_convs should handle the combined channels if needed.
                # For now, let's assume direct addition is possible if dimensions match, or use a 1x1 conv to align channels.
                # The current_dim_decoder is the target dimension for the decoder stage.
                # The conv_skip from ConvEncoder will have `conv_dims[self.num_layers - 1 - i]` channels.
                # The swin_skip (after RCASC) will have `current_dim_decoder` channels.
                # The upsampled_x will also have `current_dim_decoder` channels.
                # So, we need to ensure conv_skip matches `current_dim_decoder` before addition.
                self.fusion_convs.append(nn.Conv2d(conv_dims[self.num_layers - 1 - i], current_dim_decoder, kernel_size=1))
            else:
                self.fusion_convs.append(nn.Identity()) # No fusion conv for the first decoder stage

            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                           self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths_decoder[self.num_layers - 1 - i_layer],
                                         num_heads=num_heads[self.num_layers - 1 - i_layer],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.decoder_layers.append(layer_up)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim) # Final normalization before final upsample
        self.final_upsample = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                                  dim_scale=4, dim=embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Dual Encoders
        conv_features = self.conv_encoder(x) # List of 2D features
        swin_x, swin_downsample_features = self.forward_swin_encoder(x) # swin_x is bottleneck, swin_downsample_features are list of 1D features

        # Bottleneck
        bottleneck_output = self.conv_bottleneck(swin_x)

        # Decoder
        decoder_output = self.forward_decoder(bottleneck_output, swin_downsample_features, conv_features)

        # Final Output
        out = self.final_upsample(decoder_output)
        # The output of FinalPatchExpand_X4 is (B, L, C_out), need to reshape to (B, C_out, H, W)
        B, L, C_out = out.shape
        H_out = self.patches_resolution[0] * 4 # Assuming 4x upsampling from embed_dim stage
        W_out = self.patches_resolution[1] * 4
        out = out.view(B, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()
        out = self.output(out)
        return out

    def forward_swin_encoder(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        downsample_features = []
        for i, layer in enumerate(self.swin_encoder_layers):
            # Store feature before downsampling (skip connection)
            downsample_features.append(x)
            x = layer(x)
            
            # Apply CBAM after each Swin block output (except the last one which is bottleneck)
            if i < self.num_layers - 1: # CBAM is applied to the output of each Swin stage before downsampling
                B, L, C = x.shape
                H_res = self.patches_resolution[0] // (2 ** (i + 1))
                W_res = self.patches_resolution[1] // (2 ** (i + 1))
                x_2d = x.view(B, H_res, W_res, C).permute(0, 3, 1, 2).contiguous()
                x_2d = self.cbam_modules[i](x_2d) # Apply CBAM to 2D feature map
                x = x_2d.flatten(2).transpose(1, 2).contiguous() # Convert back to 1D for next Swin block

        x = self.norm(x) # Normalize bottleneck output
        B, L, C = x.shape
        H_bottleneck = self.patches_resolution[0] // (2 ** (self.num_layers - 1))
        W_bottleneck = self.patches_resolution[1] // (2 ** (self.num_layers - 1))
        x = x.view(B, H_bottleneck, W_bottleneck, C).permute(0, 3, 1, 2).contiguous() # Convert to 2D for ConvBottleneck
        return x, downsample_features

    def forward_decoder(self, x, swin_skips, conv_skips):
        # x is the bottleneck output (B, C, H, W) from ConvBottleneck
        # swin_skips are list of 1D features (B, L, C) from Swin Encoder
        # conv_skips are list of 2D features (B, C, H, W) from Conv Encoder

        # The first decoder layer upsamples the bottleneck output
        x = x.flatten(2).transpose(1, 2).contiguous() # Convert bottleneck output to 1D (B, L, C)

        for i, layer_up in enumerate(self.decoder_layers):
            if i > 0: # For subsequent decoder stages, perform fusion
                # Get skip features from Swin and Conv encoders
                # Swin skip features are in reverse order (deepest to shallowest)
                swin_skip_1d = swin_skips[self.num_layers - 1 - i] # (B, L, C)
                conv_skip_2d = conv_skips[self.num_layers - 1 - i] # (B, C, H, W)

                # Convert swin_skip to 2D for RCASC and fusion
                B, L_swin, C_swin = swin_skip_1d.shape
                H_swin = self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i))
                W_swin = self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i))
                swin_skip_2d = swin_skip_1d.view(B, H_swin, W_swin, C_swin).permute(0, 3, 1, 2).contiguous()

                # Apply RCASC to Swin skip
                rcasc_swin_skip = self.rcasc_modules[i-1](swin_skip_2d) # RCASC is applied to 2D feature

                # Adjust conv_skip channels if necessary and add to rcasc_swin_skip
                fused_skip = rcasc_swin_skip + self.fusion_convs[i](conv_skip_2d)

                # Upsample x to match skip connection resolution and convert to 2D
                B_x, L_x, C_x = x.shape
                H_x = self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i))
                W_x = self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i))
                x_2d = x.view(B_x, H_x, W_x, C_x).permute(0, 3, 1, 2).contiguous()

                # Add fused skip to upsampled x
                x_2d = x_2d + fused_skip
                x = x_2d.flatten(2).transpose(1, 2).contiguous() # Convert back to 1D for next decoder block

            x = layer_up(x)

        x = self.norm_up(x)
        return x

# Wrapper for compatibility with original training script
class RST_SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=9, zero_head=False, vis=False):
        super(RST_SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.rst_net = RSTnet(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                depths_decoder=config.MODEL.SWIN.DEPTHS_DECODER, # Added depths_decoder
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.rst_net(x)
        return logits

    def load_from(self, config):
        # Weight loading logic can be adapted here if needed
        print("Weight loading from pretrained Swin-T not implemented for RSTnet in this script.")
        pass

