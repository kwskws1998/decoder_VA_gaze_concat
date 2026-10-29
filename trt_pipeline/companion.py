"""Fixed native-BF16 two-view MSE training with Bvalidation-only selection."""
from contextlib import contextmanager
from datetime import datetime,timezone
import gc,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import TrainerCallback,set_seed
from trt_pipeline.data import contract_sha256,frame_sha256
from trt_pipeline.features import extract_features,feature_fingerprint,save_feature_cache
from trt_pipeline.precision import audit_model_precision,build_bf16_model,validate_saved_precision
from trt_pipeline.progress import ProgressTracker,atomic_write_json
from trt_pipeline.helpers import baseline_arguments,score,validate_feature_rows,write_prediction_table
from trt_pipeline.two_view import NOISE_STD_FRACTION,TwoViewVAEncoder,_check_partitions,_encoder_metrics,_training_device
from trt_pipeline.protocol import _validate_config,assert_empty_output,build_plan,source_hashes,pinned_roles
from trt_pipeline.companion_trainer import CompanionTrainer,optimizer_audit,serialized_optimizer_audit,unwrap_base
from trt_pipeline import training as baseline
from trt_pipeline.core.dataset import TokenizedVADataset,VABatchCollator,load_auto_tokenizer
from trt_pipeline.core.gaze import et2_feature_indices_from_names
ROOT=Path(__file__).resolve().parents[1]


def finish_companion(result, outer, output_dir, preflight=False):
    """Verify selected-model reload, export aligned mixture inputs and retain only owned duplicates."""
    from trt_pipeline.precision import load_saved_bf16_model
    from trt_pipeline.retention import retain_checkpoints
    output_dir = Path(output_dir)
    model, tokenizer = load_saved_bf16_model(result['final_model_dir'])
    model.to('cuda').eval()
    actual = extract_features(model, tokenizer, outer.iloc[:16], max_length=200, batch_size=8,
                              device=torch.device('cuda'), compute_dtype=torch.bfloat16)
    np.testing.assert_array_equal(actual['predictions'], result['outer_predictions'][:len(actual['predictions'])])
    np.testing.assert_array_equal(result['outer_labels'], outer[['valence','arousal']].to_numpy(np.float32))
    atomic_write_json(output_dir/'strict_reload.json', {'passed': True, 'actual_samples': len(actual['predictions']), 'max_abs_difference': 0})
    np.savez_compressed(output_dir/'predictions.npz', predictions=result['outer_predictions'], labels=result['outer_labels'],
                        row_ids=outer['index'].astype(str).to_numpy(dtype=str))
    retain_checkpoints(output_dir, output_dir/'final_model', result['evidence']['weights_sha256'], 2 if preflight else 10, 'companion')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result['evidence']

def _storage_precision(audit):
    """Compare stored precision independently of whether a Trainer retained gradient buffers."""
    return {key: value for key, value in audit.items() if key != 'gradient_dtypes'}

def _require_native_precision(audit):
    """Require the pinned current base's actual 334 tensors rather than a precision label."""
    if len(audit.get('parameter_dtypes', {})) != 334:
        raise ValueError('The pinned current BF16 base must have exactly 334 parameter tensors.')
    if audit.get('parameter_tensors_by_dtype') != {'bfloat16': 320, 'float32': 14}:
        raise ValueError('The actual current base requires 320 BF16 and 14 FP32 parameter tensors.')

def _validate_stage(config, train_frame, validation_frame, outer_frame, condition, budget, eval_interval, seed, preflight):
    """Reject changed main roles or budgets before any model asset or CUDA allocation."""
    _validate_config(config)
    if condition not in config.conditions:
        raise ValueError('Unknown follow-up condition.')
    if type(seed) is not int or seed != config.seed:
        raise ValueError('Each follow-up stage uses literal seed1.')
    if type(preflight) is not bool:
        raise ValueError('preflight must be an explicit boolean.')
    expected_budget, expected_interval = (2, 1) if preflight else (config.optimizer_updates, config.eval_interval)
    budget = expected_budget if budget is None else budget
    eval_interval = expected_interval if eval_interval is None else eval_interval
    if type(budget) is not int or type(eval_interval) is not int:
        raise ValueError('Stage optimizer/evaluation budgets must be literal integers.')
    if (budget, eval_interval) != (expected_budget, expected_interval):
        raise ValueError('Only the fixed main budget or explicit two-update/two-evaluation preflight is permitted.')
    _check_partitions(train_frame, validation_frame, outer_frame)
    if not preflight:
        for role, frame in (('A', train_frame), ('B_validation', validation_frame), (f'outer_fold{config.fold}', outer_frame)):
            expected_rows, expected_hash = pinned_roles(config.fold)[role]
            if len(frame) != expected_rows or frame_sha256(frame) != expected_hash:
                raise ValueError(f'Main stage rows do not match the pinned {role} frame.')
    else:
        if len(train_frame) < config.baseline.train_batch_size:
            raise ValueError('Preflight needs at least one real physical training batch of sixteen rows.')
        approved = build_plan(config)['frames']
        for role, frame in (('A', train_frame), ('B_validation', validation_frame), ('B_fit', outer_frame)):
            reference = approved[role]
            reference_ids = reference['index'].astype(str)
            positions = reference_ids.reset_index(drop=True).rename('row_id').to_frame().set_index('row_id')
            lookup = {identity: position for position, identity in enumerate(positions.index)}
            requested = frame['index'].astype(str).tolist()
            if any((identity not in lookup for identity in requested)):
                raise ValueError(f'Preflight rows must be an exact subset of approved {role}; actual outer rows are forbidden.')
            expected = reference.iloc[[lookup[identity] for identity in requested]]
            if frame_sha256(frame) != frame_sha256(expected):
                raise ValueError(f'Preflight labels/text/source changed within approved {role} row identities.')
    return (budget, eval_interval)

def _train_sampling_strategy(arguments):
    """Read the effective shuffle recipe through the same old/new HF API distinction as the factory."""
    if hasattr(arguments, 'train_sampling_strategy'):
        value = arguments.train_sampling_strategy
        strategy = getattr(value, 'value', value)
        if strategy not in ('random', 'group_by_length'):
            raise RuntimeError('Trainer exposes an unknown train_sampling_strategy.')
        return strategy
    if hasattr(arguments, 'group_by_length') and type(arguments.group_by_length) is bool:
        return 'group_by_length' if arguments.group_by_length else 'random'
    raise RuntimeError('Trainer exposes neither supported training-sampler configuration API.')

@contextmanager
def _cuda_phase(name, device, phases):
    """Measure synchronized wall/event intervals and per-phase peaks without discarding earlier peaks."""
    if name in phases:
        raise ValueError(f'Duplicate CUDA phase name: {name}')
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    first = torch.cuda.Event(enable_timing=True)
    last = torch.cuda.Event(enable_timing=True)
    first.record()
    record = {'status': 'running', 'measurement': 'actual CUDA, synchronized interval; not a utilization estimate'}
    phases[name] = record
    try:
        yield
    except BaseException:
        record['status'] = 'failed'
        raise
    else:
        record['status'] = 'completed'
    finally:
        active_failure = record['status'] == 'failed'
        try:
            last.record()
            torch.cuda.synchronize(device)
            record.update({'elapsed_seconds': time.monotonic() - start, 'cuda_event_interval_seconds': first.elapsed_time(last) / 1000, 'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)), 'peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device)), 'ending_allocated_bytes': int(torch.cuda.memory_allocated(device)), 'ending_reserved_bytes': int(torch.cuda.memory_reserved(device))})
        except Exception as error:
            record.update(elapsed_seconds=time.monotonic() - start, measurement_error=f'{type(error).__name__}: {error}')
            if not active_failure:
                raise

def _gpu_evidence(phases):
    """Retain the maximum of every measured phase after later extraction resets the counters."""
    peaks = {}
    for name in ('peak_allocated_bytes', 'peak_reserved_bytes'):
        values = [phase[name] for phase in phases.values() if name in phase]
        peaks['stage_' + name] = max(values) if values else None
        peaks[name] = peaks['stage_' + name]
        peaks[name.replace('_bytes', '_gib')] = None if not values else max(values) / 1024 ** 3
    return {'phases': dict(phases), **peaks}

def train_encoder_stage(config, train_frame, validation_frame, outer_frame, *, output_dir, condition, budget=None, eval_interval=None, seed=1, status_path=None, preflight=False):
    """Fit one fresh encoder, select only on Bvalidation, and evaluate selected outer rows once."""
    from trt_pipeline.historical.recipe import verify_runtime
    verify_runtime()
    budget, eval_interval = _validate_stage(config, train_frame, validation_frame, outer_frame, condition, budget, eval_interval, seed, preflight)
    device = _training_device(config.encoder_config())
    output_dir = Path(output_dir).expanduser().absolute()
    assert_empty_output(output_dir)
    status_path = output_dir / 'status.json' if status_path is None else Path(status_path)
    started = time.monotonic()
    phases = {}
    contract_hash = None
    base_model = wrapper = trainer = tracker = None
    events, gradient_checks, optimizer_steps, checkpoint_audits = ([], [], [], [])
    optimizer_history = []
    try:
        contract = {'schema_version': 1, 'condition': condition, 'config': config.to_dict(), 'actual_seed': seed, 'preflight_only': preflight, 'preflight_is_efficacy_evidence': False, 'train_sha256': frame_sha256(train_frame), 'validation_sha256': frame_sha256(validation_frame), 'outer_sha256': frame_sha256(outer_frame), 'optimizer_updates': budget, 'eval_interval': eval_interval, 'checkpoint_selection': 'minimum clean B_validation mse_mean', 'outer_selection': 'never; clean final selected-model evaluation only', 'outer_role': 'B_fit pseudo-outer in preflight' if preflight else f'outer Fold{config.fold} development comparison', 'two_view_noise_std_fraction': NOISE_STD_FRACTION, 'noise_scale': 'detached per-example valid-token embedding RMS; unchanged frozen TRT', 'two_view_order': ['clean', 'noisy'], 'source_sha256': source_hashes(ROOT), 'official_repositories': config.to_dict()['official_repositories'], 'training_batch_recipe': {'physical_batch': config.baseline.train_batch_size, 'accumulation': config.baseline.gradient_accumulation_steps, 'group_by_length': config.baseline.group_by_length, 'drop_last': False}}
        contract_hash = contract_sha256(contract)
        atomic_write_json(output_dir / 'contract.json', contract)
        args = baseline_arguments(config.encoder_config(), seed)
        set_seed(seed)
        with _cuda_phase('initialization_and_initial_training_probe', device, phases):
            tokenizer = load_auto_tokenizer(args.model_id, revision=args.model_revision, use_fast=True, trust_remote_code=False)
            tokenizer.padding_side = 'right'
            base_model = build_bf16_model(tokenizer, model_id=args.model_id, model_revision=args.model_revision, finetuning_mode=args.finetuning_mode, gaze_fusion=args.gaze_fusion, et_repo_id=args.et_model_id, et_revision=args.et_revision, et_filename=args.et_filename, et_cache_size=args.et_cache_size, gaze_feature_indices=et2_feature_indices_from_names(args.gaze_features), dtype=torch.bfloat16, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, attn_implementation=args.attn_implementation).to(device)
            initial_precision = audit_model_precision(base_model)
            _require_native_precision(initial_precision)
            atomic_write_json(output_dir / 'precision_initial.json', initial_precision)
            probe = train_frame.iloc[:min(128, len(train_frame))].copy()
            initial = extract_features(base_model, tokenizer, probe, max_length=args.max_length, batch_size=args.eval_batch_size, device=device, compute_dtype=torch.bfloat16)
            validate_feature_rows(initial, probe)
            initial_probe_mse = float(np.mean((initial['labels'] - initial['predictions']) ** 2))
            del initial
        wrapper = TwoViewVAEncoder(base_model)
        training_args = baseline._training_arguments(args, output_dir, bf16=True, fp16=False, fold_seed=seed)
        training_args.max_steps = budget
        training_args.eval_strategy = 'steps'
        training_args.save_strategy = 'steps'
        training_args.eval_steps = eval_interval
        training_args.save_steps = eval_interval
        training_args.data_seed = seed
        training_args.metric_for_best_model = 'mse_mean'
        training_args.greater_is_better = False
        training_args.load_best_model_at_end = True
        training_args.max_grad_norm = config.max_gradient_norm
        training_args.dataloader_drop_last = False
        actual_sampling = _train_sampling_strategy(training_args)
        if training_args.per_device_train_batch_size != config.baseline.train_batch_size or training_args.gradient_accumulation_steps != config.baseline.gradient_accumulation_steps or actual_sampling != ('group_by_length' if config.baseline.group_by_length else 'random'):
            raise RuntimeError('Trainer arguments changed the common physical batch/shuffle recipe.')

        class StageCallback(TrainerCallback):

            def on_train_begin(self, arguments, state, control, **kwargs):
                """Check the exact stored model map and publish the fixed shared/private partition."""
                current = audit_model_precision(unwrap_base(kwargs['model']))
                _require_native_precision(current)
                if _storage_precision(current) != _storage_precision(initial_precision):
                    raise RuntimeError('Trainer changed the native BF16 base storage contract.')
                atomic_write_json(output_dir / 'precision_at_train_begin.json', current)
                partition = [{'name': name, 'shape': list(parameter.shape), 'dtype': str(parameter.dtype), 'elements': parameter.numel(), 'shared': shared} for (name, parameter), shared in zip(trainer.named_base_parameters, trainer.shared_flags)]
                atomic_write_json(output_dir / 'gradient_partition.json', partition)

            def on_pre_optimizer_step(self, arguments, state, control, **kwargs):
                """Require finite post-clip model gradients."""
                gradients = [parameter.grad for parameter in kwargs['model'].parameters() if parameter.grad is not None]
                if not gradients:
                    raise RuntimeError('The encoder optimizer has no gradients.')
                norm = torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=True, foreach=True)
                if state.global_step < 2 or (state.global_step + 1) % args.logging_steps == 0:
                    event = {'optimizer_step': int(state.global_step + 1), 'finite': True, 'post_clip_total_gradient_norm': float(norm)}
                    gradient_checks.append(event)

            def on_step_end(self, arguments, state, control, **kwargs):
                """Audit actual moment tensors at updates one/two and refresh measured progress/ETA."""
                if state.global_step in (1, 2):
                    audit = optimizer_audit(kwargs['optimizer'], trainer.parameter_names())
                    if audit['initialized_parameters'] < 1:
                        raise RuntimeError('No actual optimizer moment state was initialized.')
                    optimizer_steps.append({'optimizer_step': int(state.global_step), 'audit': audit})
                    atomic_write_json(output_dir / 'optimizer_step_audits.json', optimizer_steps)
                    atomic_write_json(output_dir / 'optimizer_initial_checks.json', [{'step': event['optimizer_step'], 'live': event['audit']} for event in optimizer_steps])
                if state.global_step == 1 or state.global_step % args.logging_steps == 0 or state.global_step == budget:
                    event = tracker.update(int(state.global_step), condition=condition, seed=seed)
                    atomic_write_json(status_path, {'phase': 'training', 'condition': condition, 'seed': seed, 'run': event, 'stage_elapsed_seconds': time.monotonic() - started})

            def on_evaluate(self, arguments, state, control, metrics=None, **kwargs):
                """Keep all six clean Bvalidation metrics and the actual step/epoch for every selection event."""
                event = {'step': int(state.global_step), 'global_step': int(state.global_step), 'epoch': state.epoch, 'partition': 'Bval', 'condition': condition, 'seed': seed, **(metrics or {})}
                events.append(event)
                atomic_write_json(output_dir / 'validation_history.json', baseline._json_ready(events))
                print(json.dumps(baseline._json_ready(event), sort_keys=True), flush=True)

            def on_save(self, arguments, state, control, **kwargs):
                """Audit saved base/optimizer/loss bytes after HF writes the checkpoint, retaining compact receipts."""
                path = Path(arguments.output_dir) / f'checkpoint-{state.global_step}'
                saved = validate_saved_precision(unwrap_base(kwargs['model']), path / 'model.safetensors')
                _require_native_precision(saved)
                live = optimizer_audit(kwargs['optimizer'], trainer.parameter_names())
                serialized = serialized_optimizer_audit(path / 'optimizer.pt', kwargs['optimizer'], trainer.parameter_names())
                event = {'optimizer_step': int(state.global_step), 'checkpoint': str(path), 'precision': saved, 'optimizer_live': live, 'optimizer_serialized': serialized}
                checkpoint_audits.append(event)
                optimizer_history.append({'step': int(state.global_step), 'live': live, 'serialized': serialized})
                atomic_write_json(path / 'precision_audit.json', saved)
                atomic_write_json(output_dir / 'optimizer_checkpoint_audits.json', checkpoint_audits)
                atomic_write_json(output_dir / 'optimizer_state_history.json', optimizer_history)
        trainer = CompanionTrainer(model=wrapper, condition=condition, companion_config=config, args=training_args, train_dataset=TokenizedVADataset(train_frame, tokenizer, max_length=args.max_length), eval_dataset=TokenizedVADataset(validation_frame, tokenizer, max_length=args.max_length), data_collator=VABatchCollator(tokenizer, pad_to_multiple_of=8), processing_class=tokenizer, compute_metrics=_encoder_metrics, callbacks=[StageCallback()])
        set_seed(seed)
        tracker = ProgressTracker(budget, f'seed{seed}/{condition}', unit='optimizer_steps', event_path=output_dir / 'progress.jsonl')
        with _cuda_phase('training_periodic_validation_and_checkpoint_audits', device, phases):
            result = trainer.train()
        tracker.finish()
        expected_steps = list(range(eval_interval, budget + 1, eval_interval))
        if int(trainer.state.global_step) != budget or [event['step'] for event in events] != expected_steps or [event['optimizer_step'] for event in checkpoint_audits] != expected_steps:
            raise RuntimeError('Actual optimizer/validation/checkpoint counts differ from the fixed budget.')
        trainer.state.save_to_json(str(output_dir / 'trainer_state.json'))
        best_path = Path(trainer.state.best_model_checkpoint)
        selected = [audit for audit in checkpoint_audits if Path(audit['checkpoint']) == best_path]
        if len(selected) != 1:
            raise RuntimeError('The selected checkpoint has no unique persisted optimizer/precision receipt.')
        final_dir = output_dir / 'final_model'
        with _cuda_phase('selected_checkpoint_final_save', device, phases):
            trainer._save(final_dir)
            saved_precision = validate_saved_precision(base_model, final_dir / 'model.safetensors')
            _require_native_precision(saved_precision)
            if saved_precision['weights_sha256'] != selected[0]['precision']['weights_sha256']:
                raise RuntimeError('The final saved base is not byte-identical to the Bvalidation-selected checkpoint.')

        def extraction(frame, name):
            """Publish measured throughput/ETA and a distinct synchronized peak for each final pass."""
            extraction_tracker = ProgressTracker(len(frame), f'{condition}/{name}', event_path=output_dir / 'progress.jsonl')
            last_report = time.monotonic()

            def report(done, total, elapsed):
                """Update user-visible progress without changing the model or random state."""
                nonlocal last_report
                now = time.monotonic()
                if done == total or now - last_report >= 10:
                    event = extraction_tracker.update(done, partition=name)
                    atomic_write_json(status_path, {'phase': 'feature_extraction', 'partition': name, 'condition': condition, 'run': event, 'stage_elapsed_seconds': now - started})
                    last_report = now
            with _cuda_phase(name, device, phases):
                arrays = extract_features(base_model, tokenizer, frame, max_length=args.max_length, batch_size=args.eval_batch_size, device=device, compute_dtype=torch.bfloat16, progress=report)
            validate_feature_rows(arrays, frame)
            extraction_tracker.finish()
            return arrays
        trained_probe = extraction(probe, 'trained_training_probe')
        trained_probe_mse = float(np.mean((trained_probe['labels'] - trained_probe['predictions']) ** 2))
        del trained_probe
        validation = extraction(validation_frame, 'Bval_selected_model')
        validation_metrics = score(validation['labels'], validation['predictions'])
        validation_metrics['rmse_mean'] = (validation_metrics['rmse_valence'] + validation_metrics['rmse_arousal']) / 2
        constant = np.repeat(train_frame[['valence', 'arousal']].mean().to_numpy()[None], len(validation_frame), axis=0)
        constant_mse = float(np.mean((validation['labels'] - constant) ** 2))
        guard = {'initial_training_probe_mse': initial_probe_mse, 'trained_training_probe_mse': trained_probe_mse, 'Bval_training_mean_predictor_mse': constant_mse, 'Bval_mse_mean': validation_metrics['mse_mean'], 'finite_gradient_check_every_optimizer_step': True, 'enforced_learning_guard': not preflight, 'passed': trained_probe_mse < initial_probe_mse and validation_metrics['mse_mean'] < constant_mse, 'two_step_preflight_is_learning_evidence': False}
        evidence = {'contract_sha256': contract_hash, 'actual_seed': seed, 'condition': condition, 'preflight_only': preflight, 'weights_sha256': saved_precision['weights_sha256'], 'parameter_precision': initial_precision, 'saved_precision': saved_precision, 'best_checkpoint': str(best_path), 'selected_checkpoint_bytes_verified': True, 'best_Bval_mse_mean': trainer.state.best_metric, 'checkpoint_selection': contract['checkpoint_selection'], 'actual_optimizer_updates': int(trainer.state.global_step), 'actual_validation_events': len(events), 'actual_checkpoint_events': len(checkpoint_audits), 'validation_partition': 'B_validation', 'actual_train_sampling_strategy': actual_sampling, 'initial_batch_fingerprints': list(getattr(trainer, 'initial_batch_fingerprints', [])), 'train_metrics': baseline._json_ready(result.metrics), 'validation_metrics': validation_metrics, 'normal_learning_guard': guard, 'gradient_checks': gradient_checks, 'optimizer_step_audits': optimizer_steps, 'optimizer_checkpoint_audits': checkpoint_audits, 'phase_measurements': dict(phases), 'gpu_memory': _gpu_evidence(phases), 'timing_definitions': {'stage_elapsed_seconds': 'all initialization, probes, train/periodic validation/save/audits, final save/extraction', 'training_phase_elapsed_seconds': 'synchronized train() including periodic validation/save and optimizer audits', 'cuda_event_interval_seconds': 'CUDA stream interval including intervening CPU work; not kernel utilization', 'peak': 'maximum of separately reset phase peaks; no extrapolation'}}
        atomic_write_json(output_dir / 'gradient_checks.json', gradient_checks)
        if not preflight and (not guard['passed']):
            evidence['elapsed_seconds'] = time.monotonic() - started
            atomic_write_json(output_dir / 'failed_learning.json', evidence)
            raise RuntimeError(f'The encoder did not pass the training-only learning guard: {guard}')
        identity = {'weights_sha256': saved_precision['weights_sha256'], 'stage_contract_sha256': contract_hash}
        validation_fingerprint = feature_fingerprint(validation_frame, model_fingerprint=identity, max_length=args.max_length, compute_dtype=torch.bfloat16, tokenizer_fingerprint=args.model_revision)
        save_feature_cache(output_dir / 'validation_features.npz', validation, fingerprint=validation_fingerprint)
        outer = extraction(outer_frame, 'pseudo_outer_final_once' if preflight else 'outer_final_once')
        outer_fingerprint = feature_fingerprint(outer_frame, model_fingerprint=identity, max_length=args.max_length, compute_dtype=torch.bfloat16, tokenizer_fingerprint=args.model_revision)
        save_feature_cache(output_dir / 'outer_features.npz', outer, fingerprint=outer_fingerprint)
        write_prediction_table(outer_frame, outer['labels'], outer['predictions'], config.fold, output_dir / 'predictions.tsv')
        evidence.update({'outer_metrics': score(outer['labels'], outer['predictions']), 'outer_fingerprint': outer_fingerprint, 'validation_fingerprint': validation_fingerprint, 'phase_measurements': dict(phases), 'gpu_memory': _gpu_evidence(phases), 'elapsed_seconds': time.monotonic() - started, 'completed_utc': datetime.now(timezone.utc).isoformat()})
        atomic_write_json(output_dir / 'gpu_phases.json', _gpu_evidence(phases))
        atomic_write_json(output_dir / 'complete.json', evidence)
        atomic_write_json(status_path, {'phase': 'complete', 'condition': condition, 'elapsed_seconds': evidence['elapsed_seconds'], 'run': tracker.snapshot(), 'validation_metrics': validation_metrics, 'outer_metrics': evidence['outer_metrics']})
        return {'final_model_dir': str(final_dir), 'evidence': evidence, 'validation_metrics': validation_metrics, 'outer_predictions': outer['predictions'], 'outer_labels': outer['labels'], 'outer_row_ids': outer['row_ids'], 'outer_arrays': outer, 'validation_arrays': validation}
    except Exception as error:
        failure = {'phase': 'failed', 'condition': condition, 'error_type': type(error).__name__, 'error': str(error), 'elapsed_seconds': time.monotonic() - started, 'contract_sha256': contract_hash, 'gpu_memory': _gpu_evidence(phases)}
        atomic_write_json(output_dir / 'failed.json', failure)
        atomic_write_json(output_dir / 'gpu_phases.json', _gpu_evidence(phases))
        atomic_write_json(status_path, failure)
        if tracker is not None and tracker.status == 'running':
            tracker.finish('failed', error_type=type(error).__name__)
        raise
    finally:
        trainer = wrapper = base_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
