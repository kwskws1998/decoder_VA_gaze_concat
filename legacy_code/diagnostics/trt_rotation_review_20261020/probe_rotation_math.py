import json
import torch
from decoder_va.gaze_relation import TRTRelativeRotation
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

def explicit_rotation(x, phases):
    pairs = torch.stack([x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]], dim=-1)
    cos, sin = phases.cos(), phases.sin()
    matrices = torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)
    result = torch.einsum('bhlnc,bhlcd->bhlnd', pairs, matrices.transpose(-2, -1))
    return torch.cat([result[..., 0], result[..., 1]], dim=-1)

torch.set_num_threads(1)
all_results=[]
for fraction in (.25,.5,1.):
  for alpha_value in (0.,.15,1.):
    torch.manual_seed(172)
    rawq=torch.randn(2,4,7,256)
    rawk=torch.randn(2,2,7,256)
    rotarydim=int(256*fraction)
    angles=torch.randn(2,7,rotarydim//2)
    cos=torch.cat([angles.cos(),angles.cos()],-1)
    sin=torch.cat([angles.sin(),angles.sin()],-1)
    q,k=apply_rotary_pos_emb(rawq,rawk,cos,sin)
    q=q.detach().requires_grad_();k=k.detach().requires_grad_()
    gaze=torch.tensor([[0.,1.,3.,8.,2.,5.,4.],[2.,3.,1.,9.,0.,7.,4.]],requires_grad=True)
    mask=torch.tensor([[0,0,1,1,1,0,1],[0,1,0,1,1,1,0]],dtype=torch.bool)
    indices=torch.tensor([[2,3,4,6],[1,3,4,5]])
    compact=TRTRelativeRotation(4)
    compact.alpha.data.copy_(torch.tensor([alpha_value,-alpha_value,alpha_value*.5,alpha_value*.9]))
    actual=compact(q,k,gaze,mask,anchor_indices=indices,anchor_mask=torch.ones_like(indices,dtype=torch.bool))
    q2=q.detach().double().requires_grad_();k2=k.detach().double().requires_grad_();g2=gaze.detach().double().requires_grad_()
    a2=compact.alpha.detach().double().requires_grad_()
    kexp=k2.repeat_interleave(2,1)
    phase=a2[None,:,None]*g2[:,None,:]
    qrot=explicit_rotation(q2,phase);krot=explicit_rotation(kexp,phase)
    expected=(qrot@krot.transpose(-1,-2)-q2@kexp.transpose(-1,-2))/16
    pairs=mask[:,None,:,None]&mask[:,None,None,:]
    pairs=pairs & ~torch.eye(7,dtype=torch.bool)[None,None]
    expected=torch.where(pairs,expected,0.)
    target=torch.randn_like(actual)
    (actual*target).sum().backward();(expected*target).sum().backward()
    stats={'partial':fraction,'alpha':alpha_value,'output_max':(actual-expected).abs().max().item()}
    for name,prod,ref in [('q',q.grad,q2.grad),('k',k.grad,k2.grad),('g',gaze.grad,g2.grad),('alpha',compact.alpha.grad,a2.grad)]:
      stats[name+'_grad_max']=(prod-ref).abs().max().item()
    all_results.append(stats)
print(json.dumps(all_results,indent=2))
