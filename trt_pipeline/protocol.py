"""Pinned data and source contracts for the two original companion directions."""
import hashlib
import json
from pathlib import Path
from trt_pipeline.config import CompanionConfig, DATA_HASHES
from trt_pipeline.data import frame_sha256, split_experiences, text_groups, source_counts
from trt_pipeline.core.filters import load_filtered_folds


def verify_data(directory):
    """Require all three original inputs before reading either fold."""
    for name, expected in DATA_HASHES.items():
        path = Path(directory) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Missing or changed original data: ' + name)


def pinned_roles(fold):
    """Return the independently recorded row counts and fingerprints for one direction."""
    return json.loads(Path(__file__).with_name('pinned_roles.json').read_text())[str(fold)]


def _validate_config(config):
    """Permit relocation and either fold while rejecting changes to the fixed recipe."""
    if type(config) is not CompanionConfig or config != CompanionConfig(data_dir=config.data_dir, fold=config.fold):
        raise ValueError('Changed companion configuration')


def assert_empty_output(output):
    """Refuse implicit resume or overwriting an existing stage."""
    Path(output).mkdir(parents=True, exist_ok=False)


def source_hashes(root):
    """Fingerprint the current retained source package, independently of old run manifests."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((Path(root) / 'trt_pipeline').rglob('*.py'))}


def build_plan(config):
    """Recreate the measured normalized-group A/B and Bfit/Bvalidation splits."""
    _validate_config(config)
    verify_data(config.data_dir)
    folds = load_filtered_folds(config.data_dir, no_iemocap=True, sentence_only=False, exclude_dataset=[])
    training = (folds.fold2 if config.fold == 1 else folds.fold1).reset_index(drop=True)
    outer = (folds.fold1 if config.fold == 1 else folds.fold2).reset_index(drop=True)
    a, b = split_experiences(training, .2, 1)
    fit, validation = split_experiences(b, .2, 1701)
    frames = {'A': a, 'B': b, 'B_fit': fit, 'B_validation': validation, f'outer_fold{config.fold}': outer}
    expected = pinned_roles(config.fold)
    for role, frame in frames.items():
        if [len(frame), frame_sha256(frame)] != expected[role]:
            raise ValueError('Changed pinned split: ' + role)
    for first, second in [('A', 'B'), ('B_fit', 'B_validation')]:
        if set(text_groups(frames[first])) & set(text_groups(frames[second])):
            raise ValueError('Normalized training text groups overlap')
    if set(training['index'].astype(str)) & set(outer['index'].astype(str)):
        raise ValueError('Training/evaluation row identities overlap')
    summary = {role: {'rows': len(frame), 'frame_sha256': frame_sha256(frame), 'sources': source_counts(frame)}
               for role, frame in frames.items()}
    return {'frames': frames, 'summary': {'roles': summary, 'evaluation_fold': config.fold}}
