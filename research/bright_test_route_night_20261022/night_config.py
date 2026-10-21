"""Fixed diagnostic cases, source checkpoint and bounded overnight schedule."""
from pathlib import Path
SOURCE = Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/recover2100_dlhrjb52')
SOURCE_SHA = 'aec1a463e26ce8d9620863801ba44420fe951da91a6add9e222736d9a171197c'
BASE = SOURCE.parent.parent / 'SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS'
TARGET = 'turkey-earthquake_00000627'
TARGET_INDEX = 621
FORMAT = 'BRIGHT_TEST2500_NIGHT_DIAGNOSTICS_V1'
CASES = ('target_watched','prefix_passive','prefix_watched','full_test_watched')


def select_case(case,names):
    if len(names)!=878 or len(set(names))!=878 or names[TARGET_INDEX]!=TARGET:
        raise ValueError('Require the original canonical Test878 order and target index621')
    if case not in CASES:raise ValueError('Unknown fixed diagnostic case')
    mode='passive' if case=='prefix_passive' else 'watched'
    selected=[names[TARGET_INDEX]] if case=='target_watched' else names[:622] if case.startswith('prefix_') else list(names)
    return selected,mode,case=='full_test_watched'
