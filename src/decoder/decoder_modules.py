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
    """Returns a random of the hubert embeddings.

    Gets the batch size, timesteps, then calculate the max start index for each embedding.
    We create b dimensional array with random start sizes, depending on x_lengths. Then call slice_segments on that
    
    Args:
      () x: hubert embedding array [batch size, huberts]
      (array) x_lengths: Array that saves length of each hubert array without padding
      (int) segment_size: size of the segment to cut out
    
    Returns:
      (torch(array)) ret: sliced segment of hubert emebddings
      (int(array)) ids_str: start index of hubert embedding
    """
    if isinstance(x_lengths, list):
        x_lengths = torch.tensor(x_lengths, device=x.device)
    if (x.dim() == 2):
      b, t = x.size()
    elif (x.dim() == 3):
      b, _, t = x.size()

    if x_lengths is None:
      x_lengths = t
    ids_str_max = x_lengths - segment_size + 1 
    ids_str = (torch.rand([b], device=x.device) * ids_str_max.to(x.device)).to(dtype=torch.long)


    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


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