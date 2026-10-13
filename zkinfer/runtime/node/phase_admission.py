"""Backward-compatible imports for the renamed stage-admission module.

New code should import from :mod:`zkinfer.runtime.node.stage_admission`.
"""

from zkinfer.runtime.node.stage_admission import (
    StageAdmissionController,
    StageResourceHistory,
    process_tree_memory_gb,
    psutil,
)

# Compatibility aliases for older internal imports and historical tests.
ResourceAdmissionController = StageAdmissionController
HistoricalResourceProfiles = StageResourceHistory

__all__ = [
    "StageAdmissionController",
    "StageResourceHistory",
    "ResourceAdmissionController",
    "HistoricalResourceProfiles",
    "process_tree_memory_gb",
    "psutil",
]
