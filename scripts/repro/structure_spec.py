"""Preregistered capacity/order comparisons; no training or test-set selection."""
EXTRA_VARIANTS = {
    'flat_dual_adapter': ('flat_dual', True),
    'hierarchical_interleaved_adapter': ('hierarchical', True),
    'hierarchical_shuffle17_adapter': ('hierarchical', True),
    'hierarchical_shuffle29_adapter': ('hierarchical', True),
}
STRUCTURE_VARIANTS = ('hierarchical_adapter', *EXTRA_VARIANTS)
ORDER_CONTROLS = {
    'hierarchical_interleaved_adapter': ('interleaved', 0),
    'hierarchical_shuffle17_adapter': ('shuffle', 17),
    'hierarchical_shuffle29_adapter': ('shuffle', 29),
}
# ViT-B/16: 4*(512*128+128)+2 attention parameters, two 512->512
# adapter layers, and a 4*512 prompt. All five variants use this exact budget.
STRUCTURE_PARAMETER_COUNT = 790018


def order_settings(variant):
    policy, seed = ORDER_CONTROLS.get(variant, ('contiguous', 0))
    return {'policy': policy, 'seed': seed}
