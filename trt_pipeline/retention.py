"""Retain audit metadata before removing only this run's verified checkpoint duplicates."""
import json,shutil
from pathlib import Path
from trt_pipeline.io_utils import digest,write_json
from trt_pipeline.config import ROOT

def retain_checkpoints(stage,final_model,selected_hash,expected_events,kind):
    """Guard every deletion by exact ownership, final weights and completed save audits."""
    stage=Path(stage).resolve();final_model=Path(final_model).resolve();allowed=(ROOT/'results').resolve()
    if not stage.is_relative_to(allowed) or not final_model.is_relative_to(stage) or any(p.is_symlink() for p in (stage,final_model)):raise ValueError('Foreign checkpoint ownership.')
    if digest(final_model/'model.safetensors')!=selected_hash:raise ValueError('Final selected weights changed.')
    if kind=='historical':
        audits=[json.loads(p.read_text()) for p in sorted((stage/'integrity').glob('save_*.json'))]
        if len(audits)!=expected_events or any(not r['live']['passed'] or not r['cpu_reload']['passed'] or not r['saved_native_dtype_passed'] for r in audits):raise ValueError('Historical checkpoint audits incomplete.')
    else:
        audits=json.loads((stage/'optimizer_checkpoint_audits.json').read_text())
        if len(audits)!=expected_events or any(r['optimizer_live']['status']!='passed' or r['optimizer_serialized']['status']!='passed' for r in audits):raise ValueError('Companion checkpoint audits incomplete.')
    metadata=stage/'checkpoint_metadata';metadata.mkdir(exist_ok=False);removed=0;files=[]
    for checkpoint in sorted((stage/'checkpoints').glob('checkpoint-*')):
        if checkpoint.is_symlink() or checkpoint.parent.resolve()!=stage/'checkpoints':raise ValueError('Unexpected checkpoint path.')
        for path in checkpoint.rglob('*'):
            if path.is_symlink():raise ValueError('Checkpoint symlink forbidden.')
            if path.is_file():
                removed+=path.stat().st_size
                if path.suffix=='.json' and path.name not in ['tokenizer.json','tokenizer_config.json','vocab.json'] and path.stat().st_size<2*2**20:
                    target=metadata/checkpoint.name/path.relative_to(checkpoint);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        files.append(str(checkpoint));shutil.rmtree(checkpoint)
    write_json(stage/'checkpoint_retention.json',{'passed':True,'selected_final_preserved':True,'weights_sha256':selected_hash,'verified_save_events':len(audits),'removed_bytes':removed,'removed_own_checkpoint_paths':files})
