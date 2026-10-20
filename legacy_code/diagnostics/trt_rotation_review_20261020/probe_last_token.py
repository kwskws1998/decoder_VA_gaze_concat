import json
import torch
from transformers import Qwen3_5TextConfig,Qwen3_5TextModel
from decoder_va.gaze_attention import install_gaze_relative_attention

torch.set_num_threads(1)
torch.manual_seed(7)
config=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_num_value_heads=4,linear_num_key_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,layer_types=['linear_attention','full_attention'],max_position_embeddings=64,attention_dropout=0.,rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5,'mrope_section':[1,1,0]})
config._attn_implementation='eager'
model=Qwen3_5TextModel(config).eval()
install_gaze_relative_attention(model)
a=model.layers[-1].self_attn.gaze_relative_rotation.alpha
inputs={'input_ids':torch.tensor([[1,2,3,4,5,6,7]]),'attention_mask':torch.ones(1,7,dtype=torch.long),'gaze_relation_values':torch.tensor([[0.,0.,1.,0.,5.,0.,0.]]),'gaze_relation_mask':torch.tensor([[0,0,1,0,1,0,0]],dtype=torch.bool),'gaze_relation_indices':torch.tensor([[2,4]]),'gaze_relation_anchor_mask':torch.ones(1,2,dtype=torch.bool),'use_cache':False}
zero=model(**inputs).last_hidden_state.detach()
a.data.fill_(.4)
changed=model(**inputs).last_hidden_state
(changed[:,-1]*torch.randn_like(changed[:,-1])).sum().backward()
print(json.dumps({'max_any_token_change':(changed-zero).abs().max().item(),'max_pooled_last_token_change':(changed[:,-1]-zero[:,-1]).abs().max().item(),'alpha_gradient':a.grad.tolist()},indent=2))
