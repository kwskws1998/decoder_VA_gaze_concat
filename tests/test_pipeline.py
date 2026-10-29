"""Regression checks for the three retained workflows and their archived numerical operators."""
import ast
import copy
import hashlib
import io
import json
import types
import zipfile

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from trt_pipeline.config import CompanionConfig, ROOT
from trt_pipeline.mixing import mix_predictions, mix_files, fold_metric_mean, KEYS
from trt_pipeline.protocol import build_plan

ARCHIVE=ROOT/'artifacts/cleanup_20261029/original_workspace.zip'
BACKUP=ROOT/'server_backup_20261028/provenance'


def archived_nodes(path,names):
    """Load exact previous operator definitions without importing retired experiment packages."""
    with zipfile.ZipFile(ARCHIVE) as z: tree=ast.parse(z.read(path))
    return [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]


@pytest.mark.parametrize('fold,counts',[(1,[21517,5381,4305,1076,26893]),(2,[21514,5379,4302,1077,26898])])
def test_original_splits_and_data_fingerprints(fold,counts):
    """Require both original A/B/Bfit/Bval/outer partitions and their saved hashes."""
    frames=build_plan(CompanionConfig(fold=fold))['frames']
    assert [len(v) for v in frames.values()]==counts


@pytest.mark.parametrize('overrides',[{'seed':2},{'optimizer_updates':2},{'eval_interval':1},{'conditions':('bmc',)},{'fold':3},{'max_gradient_norm':2}])
def test_changed_companion_recipe_rejected(overrides):
    with pytest.raises(ValueError): CompanionConfig(**overrides)


class TinyBase(nn.Module):
    """Exercise the real two-view packing and gradient path without external assets."""
    def __init__(self):
        super().__init__();self.embeddings=nn.Embedding(12,5);self.projection=nn.Linear(5,5);self.head=nn.Linear(5,2)
        self.config=types.SimpleNamespace()
        self.gaze_provider=types.SimpleNamespace(compute=lambda ids,mask:(torch.zeros((*ids.shape,1)),mask.bool()))
        self.backbone=self.encode_backbone
    def get_input_embeddings(self):
        return self.embeddings
    def encode_backbone(self,inputs_embeds,**kwargs):
        return types.SimpleNamespace(last_hidden_state=self.projection(inputs_embeds))
    def _gaze_inputs(self,ids,mask,embeddings,gaze):
        return embeddings,mask,None,mask.sum(1)-1
    def predict_pooled(self,pooled):
        return torch.sigmoid(self.head(pooled))
    def forward(self,input_ids,attention_mask):
        hidden=self.projection(self.embeddings(input_ids));pooled=hidden[torch.arange(len(hidden)),attention_mask.sum(1)-1]
        return {'logits':self.predict_pooled(pooled)}


def test_companion_forward_loss_gradients_match_original():
    """Compare exact clean/noisy values, loss and parameter gradients with the completed operator."""
    from trt_pipeline.two_view import TwoViewVAEncoder
    from trt_pipeline.companion_trainer import CompanionTrainer
    scope={'torch':torch,'nn':nn,'NOISE_STD_FRACTION':.01}
    exec(compile(ast.Module(body=archived_nodes('va_alternatives_code/encoder_training.py',{'gaussian_text_view','TwoViewVAEncoder'}),type_ignores=[]),'archived_operator','exec'),scope)
    old=scope['TwoViewVAEncoder'](TinyBase());new=TwoViewVAEncoder(copy.deepcopy(old.base_model))
    with zipfile.ZipFile(ARCHIVE) as z: old_class=next(n for n in ast.parse(z.read('va_followup_code/trainer.py')).body if isinstance(n,ast.ClassDef) and n.name=='FollowupTrainer')
    method=next(n for n in old_class.body if isinstance(n,ast.FunctionDef) and n.name=='compute_loss')
    loss_scope={'torch':torch,'F':F};exec(compile(ast.Module(body=[method],type_ignores=[]),'archived_loss','exec'),loss_scope)
    inputs={'input_ids':torch.tensor([[1,2,3,0],[4,5,0,0]]),'attention_mask':torch.tensor([[1,1,1,0],[1,1,0,0]]),'labels':torch.tensor([[.2,.8],[.6,.4]])}
    previous=types.SimpleNamespace(bmc_loss=None,last_loss_components={})
    retained=object.__new__(CompanionTrainer);retained.last_loss_components={}
    torch.manual_seed(13);old_loss=loss_scope['compute_loss'](previous,old,inputs);old_loss.backward()
    torch.manual_seed(13);new_loss=retained.compute_loss(new,inputs);new_loss.backward()
    assert torch.equal(old_loss,new_loss)
    for a,b in zip(old.parameters(),new.parameters()):assert torch.equal(a.grad,b.grad)
    old.eval();new.eval();assert torch.equal(old(**{k:v for k,v in inputs.items() if k!='labels'})['logits'],new(**{k:v for k,v in inputs.items() if k!='labels'})['logits'])


def test_noise_mask_scale_and_detached_gradient():
    from trt_pipeline.two_view import gaussian_text_view
    x=torch.ones(256,10,32,requires_grad=True);mask=torch.ones(256,10,dtype=torch.long);mask[:,8:]=0
    torch.manual_seed(7);y=gaussian_text_view(x,mask)
    assert torch.equal(y[:,8:],x[:,8:])
    assert abs(float((y[:,:8]-x[:,:8]).detach().std())-.01)<.0002
    y.sum().backward();assert torch.equal(x.grad,torch.ones_like(x))


@pytest.mark.parametrize('fold',[1,2])
def test_saved_prediction_mixture_is_byte_exact(fold,tmp_path):
    """Recompute the completed mixture on all real rows in both original directions."""
    with zipfile.ZipFile(BACKUP/'trt_oof_code/two_fold_verified_package_20261028.zip') as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('two_fold_results_0382b3d7.zip'))) as z:
            with np.load(io.BytesIO(z.read(f'fold{fold}_predictions.npz')),allow_pickle=False) as a:
                baseline=a['predictions_baseline'].copy();labels=a['labels'].copy();ids=a['row_ids'].copy();expected=a['predictions_prediction_mixing'].copy()
            if fold==2:
                with np.load(io.BytesIO(z.read('companion/predictions.npz')),allow_pickle=False) as a:companion=a['predictions'].copy()
    if fold==1:
        with zipfile.ZipFile(BACKUP/'va_followup_code/fold1_followup_results_628a2062.zip') as z:
            with np.load(io.BytesIO(z.read('predictions.npz')),allow_pickle=False) as a:companion=a['predictions_matched_mse'].copy()
    assert np.array_equal(mix_predictions(baseline,companion),expected)
    np.savez(tmp_path/'base.npz',predictions=baseline,labels=labels,row_ids=ids)
    np.savez(tmp_path/'comp.npz',predictions=companion,labels=labels,row_ids=ids)
    metrics=mix_files(tmp_path/'base.npz',tmp_path/'comp.npz',tmp_path/'mixed.npz')
    with np.load(tmp_path/'mixed.npz',allow_pickle=False) as a:assert np.array_equal(a['predictions'],expected)
    report=json.loads((BACKUP/'trt_oof_code/final_report_20261028.json').read_text())
    for key in KEYS:assert metrics[key]==pytest.approx(report['fold_metrics'][str(fold)]['prediction_mixing'][key],abs=1e-12)


@pytest.mark.parametrize('bad',[np.array([[np.nan,.2]]),np.array([[1.1,.2]]),np.array([.1,.2]),np.zeros((1,3))])
def test_invalid_mixture_arrays_rejected(bad):
    with pytest.raises(ValueError):mix_predictions(bad,bad)


def test_reordered_prediction_rows_rejected(tmp_path):
    y=np.array([[.1,.2],[.3,.4]]);ids=np.array(['한글','second'])
    np.savez(tmp_path/'a.npz',predictions=y,labels=y,row_ids=ids)
    np.savez(tmp_path/'b.npz',predictions=y,labels=y,row_ids=ids[::-1])
    with pytest.raises(ValueError):mix_files(tmp_path/'a.npz',tmp_path/'b.npz',tmp_path/'out.npz')
    assert not (tmp_path/'out.npz').exists()


def test_fold_metrics_use_arithmetic_means():
    first={key:.2 for key in KEYS};second={key:.8 for key in KEYS}
    assert fold_metric_mean(first,second)=={key:.5 for key in KEYS}


def test_historical_snapshot_bytes_unchanged():
    root=ROOT/'trt_pipeline/historical/snapshot';manifest=json.loads((root/'source.json').read_text())
    for name,expected in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected


def test_no_retired_imports_and_only_supported_commands():
    from trt_pipeline.cli import build_parser
    retired=('va_model_code','historical_bf16_code','residual_memory_code','va_alternatives_code','va_followup_code','trt_oof_code','trt_optuna_code','trt_posttraining_code','legacy','diffusion_code','expel_code')
    for p in (ROOT/'trt_pipeline').rglob('*.py'):
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n,ast.ImportFrom):assert not (n.module or '').startswith(retired),(p,n.module)
            if isinstance(n,ast.Import):assert all(not a.name.startswith(retired) for a in n.names),p
    assert build_parser().parse_args(['companion','--fold','2']).fold==2
    for name in ['lora','weight_mixing','optuna','bmc','cagrad','diffusion']:
        with pytest.raises(SystemExit):build_parser().parse_args([name])


def test_baseline_mse_value_and_gradient_match_archived():
    """Removing unused CCC objectives must leave the measured MSE operator unchanged."""
    from dataclasses import dataclass
    from trt_pipeline.core.losses import va_regression_loss
    scope={'dataclass':dataclass,'Tensor':torch.Tensor,'F':F,'LOSS_CHOICES':('mse','ccc','mse+ccc')}
    nodes=archived_nodes('va_model_code/decoder_va/losses.py',{'LossBreakdown','concordance_correlation_coefficient','va_regression_loss'})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'archived_mse','exec'),scope)
    x=torch.tensor([[.2,.3],[.4,.9]],requires_grad=True);copy_x=x.detach().clone().requires_grad_(True);y=torch.tensor([[.1,.6],[.8,.7]])
    old=scope['va_regression_loss'](x,y,'mse').total;new=va_regression_loss(copy_x,y,'mse').total
    old.backward();new.backward();assert torch.equal(old,new) and torch.equal(x.grad,copy_x.grad)
    with pytest.raises(ValueError):va_regression_loss(copy_x,y,'ccc')


def test_full_backbone_path_matches_archived(monkeypatch):
    """Removing PEFT branches must preserve the full-model load and precision path."""
    from trt_pipeline.core import model
    base=nn.Linear(3,2).to(torch.bfloat16)
    seen=[]
    def load(*args,**kwargs):
        seen.append((args,kwargs));return copy.deepcopy(base)
    monkeypatch.setattr(model,'_load_qwen_text_backbone',load)
    scope=dict(vars(model));nodes=archived_nodes('va_model_code/decoder_va/model.py',{'load_qwen_backbone'})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'archived_backbone','exec'),scope)
    old=scope['load_qwen_backbone'](finetuning_mode='full');new=model.load_qwen_backbone(finetuning_mode='full')
    assert seen[0]==seen[1]
    for first,second in zip(old.parameters(),new.parameters()):assert torch.equal(first,second) and first.dtype==second.dtype==torch.float32
    with pytest.raises(ValueError):model.load_qwen_backbone(finetuning_mode='lora')


def test_retained_argument_normalization_and_baseline_recipe(monkeypatch):
    """Check the complete historical contract and the distinct companion batch policy."""
    from trt_pipeline.historical.train_raw import build_parser
    from trt_pipeline.historical import recipe
    from trt_pipeline.training import _validate_args
    from trt_pipeline.helpers import baseline_arguments
    args=build_parser().parse_args(['qwen3.5-0.8b','mse'])
    _validate_args(args)
    monkeypatch.setattr(recipe,'verify_runtime',lambda:None)
    recipe.validate_args(args)
    assert tuple(args.gaze_feature_indices)==(3,) and args.train_batch_size==16 and args.group_by_length
    companion=baseline_arguments(CompanionConfig().encoder_config(),1)
    assert not companion.group_by_length and companion.train_batch_size==16 and companion.gradient_accumulation_steps==1
