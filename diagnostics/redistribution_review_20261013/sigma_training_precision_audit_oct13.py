"""Reproduce parameter and output rounding under the experiment's pinned Trainer."""
import collections
import json
import tempfile
from pathlib import Path
import sys
import torch
import transformers
import accelerate
sys.path.insert(0, '/Users/wansookim/Documents/decoder_based_va_prediction')
from va_model_code.tests.test_sigma_diagnostics import make_trainer


def main():
    """Compare actual training parameter storage and constant-gradient AdamW updates."""
    result = {'versions': {'torch':torch.__version__, 'transformers':transformers.__version__, 'accelerate':accelerate.__version__}, 'scalar_updates': [], 'trainer_runs': []}
    for fused in (False, True):
        for dtype in (torch.float32, torch.bfloat16):
            parameter = torch.nn.Parameter(torch.tensor([.001,.01,.02,.1,1.],dtype=dtype))
            optimizer = torch.optim.AdamW([parameter],lr=6e-6,weight_decay=.01,fused=fused)
            before=parameter.detach().float().tolist()
            for _ in range(4490):
                optimizer.zero_grad()
                parameter.grad=torch.ones_like(parameter)
                optimizer.step()
            result['scalar_updates'].append({'fused':fused,'dtype':str(dtype),'before':before,'after':parameter.detach().float().tolist(),'moment_dtype':str(optimizer.state[parameter]['exp_avg'].dtype)})
    for upcast in (False, True):
        with tempfile.TemporaryDirectory(prefix='sigma-bf16-trainer-') as directory:
            trainer=make_trainer(directory,'fixed-gaussian',torch.bfloat16,left=.25,right=1.)
            if upcast:
                trainer.model.backbone.float()
            trainer.args.max_steps=10
            trainer.args.learning_rate=6e-6
            trainer.args.max_grad_norm=1.
            trainer.args.logging_steps=10
            trainer.args.disable_tqdm=True
            before={name:parameter.detach().clone() for name,parameter in trainer.model.named_parameters()}
            trainer.train()
            report={'backbone_upcast':upcast,'optimizer_argument':str(trainer.args.optim),'optimizer_class':type(trainer.optimizer.optimizer).__name__,'optimizer_defaults':{k:v for k,v in trainer.optimizer.defaults.items() if k not in ('params',)},'groups':{}}
            for prefix in ('backbone.','gaze_projector.','regression_head.'):
                parameters=[(name,parameter) for name,parameter in trainer.model.named_parameters() if name.startswith(prefix)]
                total=sum(parameter.numel() for _,parameter in parameters)
                changed=sum(int((parameter!=before[name]).sum()) for name,parameter in parameters)
                states=collections.Counter(str(trainer.optimizer.state[parameter]['exp_avg'].dtype) for _,parameter in parameters if parameter in trainer.optimizer.state and 'exp_avg' in trainer.optimizer.state[parameter])
                report['groups'][prefix]={'total':total,'changed':changed,'changed_percent':100*changed/total,'parameter_dtypes':sorted({str(parameter.dtype) for _,parameter in parameters}),'moment_dtypes':dict(states)}
            result['trainer_runs'].append(report)
    logits=torch.tensor([-.001,0,.001],dtype=torch.bfloat16)
    result['hardsigmoid']={'bf16_logits':logits.float().tolist(),'bf16_activation':torch.nn.functional.hardsigmoid(logits).float().tolist(),'fp32_activation':torch.nn.functional.hardsigmoid(logits.float()).tolist()}
    output=Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print(output)


if __name__=='__main__':
    main()
