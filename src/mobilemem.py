"""Shared MobileMem subset helpers."""

MOBILEMEM_SUBSETS = (
    "mobilemem_shopping_graph_hard_120",
    "mobilemem_shopping_graph_noncalc_hard_120",
)

MOBILEMEM_HIGH_NOISE_SUBSETS = (
    "mobilemem_shopping_graph_hard_120",
    "mobilemem_shopping_graph_noncalc_hard_120",
)

MOBILEMEM_GRAPH_SUBSETS = (
    "mobilemem_shopping_graph_hard_120",
    "mobilemem_shopping_graph_noncalc_hard_120",
)

_MOBILEMEM_SUBSET_SET = set(MOBILEMEM_SUBSETS)
_MOBILEMEM_HIGH_NOISE_SUBSET_SET = set(MOBILEMEM_HIGH_NOISE_SUBSETS)
_MOBILEMEM_GRAPH_SUBSET_SET = set(MOBILEMEM_GRAPH_SUBSETS)


def is_mobilemem_subset(subset: str) -> bool:
    return subset in _MOBILEMEM_SUBSET_SET


def is_mobilemem_high_noise_subset(subset: str) -> bool:
    return subset in _MOBILEMEM_HIGH_NOISE_SUBSET_SET


def is_mobilemem_graph_subset(subset: str) -> bool:
    return subset in _MOBILEMEM_GRAPH_SUBSET_SET
