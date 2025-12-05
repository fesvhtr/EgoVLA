import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(
               output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerSplitActV2(nn.Module):
  def __init__(
    self,
    hidden_size,
    proprio_size,
    out_dim,
    use_proprio,
    # use_hand_input,
    sep_proprio,
    # NOTE: Optional extra mano feature token (draft support)
    extra_mano_dim: int = 0,
    use_extra_mano_token: bool = False,
    **kwargs
  ):
    super().__init__()

    self.use_proprio = use_proprio
    self.sep_proprio = sep_proprio

    self.proprio_size = proprio_size

  # NOTE: Whether to use an additional mano token, projected by an MLP and
  # concatenated with the existing decoder input tokens.
    self.extra_mano_dim = extra_mano_dim
    self.use_extra_mano_token = use_extra_mano_token

    self.proprio_projection = nn.Sequential(
      nn.Linear(self.proprio_size, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, hidden_size)
    )

    self.proprio_projection_3d = nn.Sequential(
      nn.Linear(3, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, hidden_size)
    )
    self.proprio_projection_rot = nn.Sequential(
      nn.Linear(3, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, hidden_size)
    )
    self.proprio_projection_hand = nn.Sequential(
      nn.Linear(5 * 3, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, hidden_size)
    )

  # NOTE:Optional MLP for extra mano token (e.g., 55-dim feature)
  if self.use_extra_mano_token and self.extra_mano_dim > 0:
    self.proprio_projection_mano = nn.Sequential(
      nn.Linear(self.extra_mano_dim, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, hidden_size)
    )

    self.first_norm = nn.LayerNorm(hidden_size)
    self.out_dim = out_dim

    # 3D Translation * 2
    self.output_projection_left = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, 3 + 6 + 15)
    )

    # 3D Rotation: 6 * 2
    self.output_projection_right =  nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, 3 + 6 + 15)
    )

    num_transformer_encoder_layers = 6
    self.layers = nn.ModuleList()
    for _ in range(num_transformer_encoder_layers):
      encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_size, nhead=1, batch_first=True,
        # dropout=0
        activation=torch.nn.functional.elu,
      )
      self.layers.append(encoder_layer)

    self.layers.train()

    # Transformer
    for param in self.layers.parameters():
      param.requires_grad = True
    # Proprio Projection
    # for param in self.proprio_projection_2d.parameters():
    #   param.requires_grad = True
    for param in self.proprio_projection_3d.parameters():
      param.requires_grad = True
    for param in self.proprio_projection_rot.parameters():
      param.requires_grad = True
    for param in self.proprio_projection_hand.parameters():
      param.requires_grad = True

  # NOTE: Extra mano projection, if enabled
  if hasattr(self, "proprio_projection_mano"):
    for param in self.proprio_projection_mano.parameters():
      param.requires_grad = True

    # Output Projection
    for param in self.output_projection_left.parameters():
      param.requires_grad = True
    for param in self.output_projection_right.parameters():
      param.requires_grad = True

  def forward(
    self, latent, input_dict, memory, memory_mask,
  ):
    proprio_input = input_dict["proprio"]
    proprio_input = self.proprio_projection(proprio_input)
    proprio_input = proprio_input.unsqueeze(1)
    # Proprio input shape: (B, 1, D)
    latent = latent.reshape(
      proprio_input.shape[0],
      latent.shape[0] // proprio_input.shape[0],
      latent.shape[1]
    )

    if self.use_proprio and self.sep_proprio:
      proprio_input_3d = input_dict["proprio_3d"].reshape(-1, 2, 3)
      proprio_input_3d = self.proprio_projection_3d(proprio_input_3d)
      # proprio_input_3d = proprio_input_3d.unsqueeze(1)

      proprio_input_rot = input_dict["proprio_rot"].reshape(-1, 2, 3)
      proprio_input_rot = self.proprio_projection_rot(proprio_input_rot)
      # proprio_input_rot = proprio_input_rot.unsqueeze(1)

      proprio_input_hand = input_dict["proprio_hand_finger_tip"].reshape(-1, 2, 5 * 3)
      proprio_input_hand = self.proprio_projection_hand(proprio_input_hand)
      # proprio_input_handdof = proprio_input_handdof.unsqueeze(1)

    if self.use_proprio:
      if self.sep_proprio:
      prefix_tokens = [
        # proprio_input_2d,
        proprio_input_3d,
        proprio_input_rot,
        proprio_input_hand,
      ]
      # Optionally insert one extra mano token at the front
      if self.use_extra_mano_token and "mano_token" in input_dict and hasattr(self, "proprio_projection_mano"):
        mano_feat = input_dict["mano_token"]
        # Expect shape (B, D_mano); allow (B, 1, D_mano) as well in a loose way.
        if mano_feat.dim() == 2:
          mano_emb = self.proprio_projection_mano(mano_feat).unsqueeze(1)
        elif mano_feat.dim() == 3:
          # Flatten any temporal dimension into token dimension (draft behavior)
          B, T, D = mano_feat.shape
          mano_emb = self.proprio_projection_mano(mano_feat.reshape(B * T, D)).reshape(B, T, -1)
        else:
          mano_emb = None
        if mano_emb is not None:
          prefix_tokens = [mano_emb] + prefix_tokens

      latent = torch.cat(prefix_tokens + [latent], dim=1)
      else:
      # Simple (non-split) case: append mano token after the main proprio token.
      cat_list = [proprio_input]
      if self.use_extra_mano_token and "mano_token" in input_dict and hasattr(self, "proprio_projection_mano"):
        mano_feat = input_dict["mano_token"]
        if mano_feat.dim() == 2:
          mano_emb = self.proprio_projection_mano(mano_feat).unsqueeze(1)
        elif mano_feat.dim() == 3:
          B, T, D = mano_feat.shape
          mano_emb = self.proprio_projection_mano(mano_feat.reshape(B * T, D)).reshape(B, T, -1)
        else:
          mano_emb = None
        if mano_emb is not None:
          cat_list.append(mano_emb)
      cat_list.append(latent)
      latent = torch.cat(cat_list, dim=1)


    batch_size, latent_len, _ = latent.shape

    mask = None
    pos = None
    # output = src
    memory_mask = ~memory_mask
    # memory = memory.detach()
    memory_mask = memory_mask.detach()

    src_key_padding_mask = torch.zeros(
      latent.shape[0],
      latent.shape[1]
    ).bool().to(memory_mask.device)
    src_key_padding_mask = torch.concat([
      memory_mask, src_key_padding_mask
    ], dim=1)

    latent = self.first_norm(
      latent
    )

    memory = self.first_norm(
      memory
    )

    for layer in self.layers:
        input_latent = torch.concat([
          memory, latent
        ], dim = 1)
        latent = layer(
          # latent,
          input_latent,
          src_key_padding_mask=src_key_padding_mask,
          # pos=pos
        )
        latent = latent[:, -latent_len:]

    if self.use_proprio:
      if self.sep_proprio:
      # For split-proprio case we previously had exactly 6 prefix tokens
      # (2 for 3D, 2 for rot, 2 for hand). If we insert an extra mano token,
      # we need to skip one more token here.
      num_prefix_tokens = 6
      if self.use_extra_mano_token and "mano_token" in input_dict and hasattr(self, "proprio_projection_mano"):
        num_prefix_tokens += 1
      latent = latent[:, num_prefix_tokens:, :]
      else:
      # For non-split case we previously had 1 prefix token (proprio).
      num_prefix_tokens = 1
      if self.use_extra_mano_token and "mano_token" in input_dict and hasattr(self, "proprio_projection_mano"):
        num_prefix_tokens += 1
      latent = latent[:, num_prefix_tokens:, :]
 
    out_left = self.output_projection_left(
      latent[:, ::2]
    ).reshape(
      -1, 1, 3 + 6 + 15
    )
    out_right = self.output_projection_right(
      latent[:, 1::2]
    ).reshape(
      -1, 1, 3 + 6 + 15
    )

    output = torch.cat([
      out_left, out_right
    ], dim=1).reshape(-1, 2 * (3 + 6 + 15))


    return {
      "pred": output
    }

  def inference(
    self, latent, input_dict, memory, memory_mask,
    x=None, return_kl=False
  ):
    return self.forward(
      latent, input_dict, memory, memory_mask
    )
