import torch

# Slice Segment function of VITS
def slice_segments(x, ids_str, segment_size=4):
    """Returns the sliced embeddings.
    
    Create new empty array. For each batch, slice  it from start to end index.
    
    Args:
      x: batch with hubert embeddings
      ids_str: random calculated start index for each embedding
      segment_size: size of the segment

    Return:
      ret: Array with all the sliced huberts embeddings
    """
    if(x.dim() == 2):
      ret = torch.zeros_like(x[:, :segment_size])
      for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]

    elif x.dim() == 3:
        ret = torch.zeros_like(x[:, :, :segment_size])
        for i in range(x.size(0)):
            idx_str = ids_str[i]
            idx_end = idx_str + segment_size
            ret[i] = x[i, :, idx_str:idx_end]
    
    return ret



def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """
    x: (B, T) or (B, T, C)  - HuBERT ids or embeddings
    x_lengths: tensor of shape (B,) with actual sequence lengths in T.
    segment_size: desired number of frames (for 2.5s @ 50Hz -> 125).

    Returns:
      x_seg: (B, segment_size, ...)  – always fixed length (padded if needed)
      start_ids: (B,) start index in original HuBERT time
    """
    if x.dim() == 3:
        B, T, C = x.shape
    elif x.dim() == 2:
        B, T = x.shape
        C = None
    else:
        raise ValueError(f"x must be 2D or 3D, got {x.shape}")

    if x_lengths is None:
        x_lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
    else:
        x_lengths = x_lengths.to(x.device)

    start_ids = []
    seg_list = []

    for i in range(B):
        L = int(x_lengths[i].item())
        if L <= 0:
            L = T

        if L >= segment_size:
            max_start = L - segment_size
            s = torch.randint(0, max_start + 1, (1,), device=x.device).item()
            seg_len = segment_size
        else:
            # sequence shorter than desired -> take from 0 and pad later
            s = 0
            seg_len = L

        if x.dim() == 3:
            seg = x[i, s:s + seg_len, :]  # (seg_len, C)
            if seg_len < segment_size:
                pad = seg.new_zeros(segment_size - seg_len, C)
                seg = torch.cat([seg, pad], dim=0)
        else:
            seg = x[i, s:s + seg_len]     # (seg_len,)
            if seg_len < segment_size:
                pad = seg.new_zeros(segment_size - seg_len)
                seg = torch.cat([seg, pad], dim=0)

        seg_list.append(seg.unsqueeze(0))
        start_ids.append(s)

    x_seg = torch.cat(seg_list, dim=0)                      # (B, segment_size, C?) or (B, segment_size)
    start_ids = torch.tensor(start_ids, device=x.device, dtype=torch.long)
    return x_seg, start_ids


def broadcast_single_embedding(hubert_emb, style_emb):
      """Function to broadcast the embedding .
      
      Broadcasts the embeddings such that we have 
      all the embedding channels per hubert timestep.
      
      Args:
        (int(array)) hubert_emb: Array containing the hubert embeddings.
        (float(array)) speaker_emb: Array containing the speaker embeddings.
        (float(array)) style_emb: Array containing the style embeddings.

      Returns:
        (tensor(array)) output: Broadcasted Tensor.
      """

      # Get the speaker and style combined and adjust shape
      emb = style_emb.unsqueeze(-1) # (Batch, 640, 1)
      emb = emb.repeat(1, 1, hubert_emb.size(-1))  # Shape: (Batch, 640, steps)


      # Adjust shape of hubert and broadcast
      # hubert_emb = hubert_emb.unsqueeze(1)  # Shape: (Batch, 1, Temporal) #? => hubert is already [batch. 128, Temp] bc of emb
      output = torch.cat([emb, hubert_emb], dim=1) # (Batch, 640 + 128 , 0 + steps) => (Batch, 768, steps)
      
      return output


def broadcast_embeddings(hubert_emb, speaker_emb, style_emb):
      """Function to broadcast the embedding .
      
      Broadcasts the embeddings such that we have 
      all the embedding channels per hubert timestep.
      
      Args:
        (int(array)) hubert_emb: Array containing the hubert embeddings.
        (float(array)) speaker_emb: Array containing the speaker embeddings.
        (float(array)) style_emb: Array containing the style embeddings.

      Returns:
        (tensor(array)) output: Broadcasted Tensor.
      """

      # Get the speaker and style combined and adjust shape
      spk_style_emb = torch.cat([speaker_emb, style_emb], dim=1)  # Shape: (Batch, 640)
      spk_style_emb = spk_style_emb.unsqueeze(-1) # (Batch, 640, 1)
      spk_style_emb = spk_style_emb.repeat(1, 1, hubert_emb.size(-1))  # Shape: (Batch, 640, steps)


      # Adjust shape of hubert and broadcast
      # hubert_emb = hubert_emb.unsqueeze(1)  # Shape: (Batch, 1, Temporal) #? => hubert is already [batch. 128, Temp] bc of emb
      output = torch.cat([spk_style_emb, hubert_emb], dim=1) # (Batch, 640 + 128 , 0 + steps) => (Batch, 768, steps)
      
      return output


def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses



import torch

def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1. / norm_type)
    return total_norm