"""Exercise structural TRT-relative restrictions on a tiny native Qwen CPU model."""
import json,torch
from transformers import Qwen3_5TextConfig,Qwen3_5TextModel
from decoder_va.gaze_attention import install_gaze_relative_attention
from decoder_va.gaze_relation import TRTRelativeRotation

def model_case(layer_types,anchors):
    torch.manual_seed(931)
    config=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=len(layer_types),num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_num_value_heads=2,linear_num_key_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,layer_types=layer_types,attention_dropout=0.,max_position_embeddings=128,rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[1,1,2]})
    config._attn_implementation='eager';m=Qwen3_5TextModel(config).eval();install_gaze_relative_attention(m)
    mask=torch.zeros(1,6,dtype=torch.bool);mask[0,anchors]=True
    values=torch.zeros(1,6);values[0,anchors]=torch.arange(len(anchors)).float()+1
    inputs={'input_ids':torch.tensor([[1,2,3,4,5,6]]),'attention_mask':torch.ones(1,6,dtype=torch.long),'use_cache':False,'gaze_relation_values':values,'gaze_relation_mask':mask}
    alpha=[p for n,p in m.named_parameters() if n.endswith('gaze_relative_rotation.alpha')]
    before=m(**inputs).last_hidden_state.detach()
    with torch.no_grad():
        for p in alpha:p.fill_(0.7)
    after=m(**inputs).last_hidden_state
    (after[:,-1]*torch.arange(32)).sum().backward()
    return {'layer_types':layer_types,'anchors':anchors,'all_token_change':(after-before).abs().max().item(),'pooled_last_token_change':(after[:,-1]-before[:,-1]).abs().max().item(),'alpha_grad_norms':[p.grad.norm().item() for p in alpha]}

def offset_case():
    torch.manual_seed(19);m=TRTRelativeRotation(2)
    with torch.no_grad():m.alpha.fill_(.4)
    q=torch.randn(1,2,5,8);k=torch.randn_like(q);v=torch.tensor([[0.,1.,2.,5.,0.]]);mask=torch.tensor([[False,True,True,True,False]])
    a=m(q,k,v,mask);b=m(q,k,v+10.,mask)
    return {'max_difference_after_common_offset10':(a-b).abs().max().item()}
print(json.dumps({'cases':[model_case(['full_attention'],[1]),model_case(['full_attention'],[1,3]),model_case(['full_attention','full_attention'],[1,3]),model_case(['full_attention'],[1,5])],'offset':offset_case()},indent=2))
