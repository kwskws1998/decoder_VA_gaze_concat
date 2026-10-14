"""Reproduce silent later-subword fallback after an invalid first ET prediction."""
import sys
sys.path.insert(0,'/Users/wansookim/Documents/decoder_based_va_prediction')
import torch
from va_model_code.tests.test_gaze import FakeET2GazeProvider,FakeTargetTokenizer,FakeETModel
class NonfiniteFirstSubword(FakeETModel):
    """Keep later ET pieces finite while invalidating Hello's first piece."""
    def forward(self,input_ids,attention_mask):
        output=super().forward(input_ids,attention_mask)
        output[input_ids==10,3]=float('nan')
        return output
provider=FakeET2GazeProvider(FakeTargetTokenizer())
provider.fake_model=NonfiniteFirstSubword()
ids=torch.tensor([[0,10,11,12,2]])
features,mask=provider.compute(ids,torch.ones_like(ids))
assert features[0,1,0].item()==11.0
assert mask[0,1].item() is True
print({'features':features[:,:,0].tolist(),'mask':mask.tolist(),'later_piece_accepted':True})
