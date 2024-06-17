tome_presets = {
    'bsm_hq': [
        dict(q_mode=None, kv_mode='bsm', kv_r=0.6, kv_sx=2, kv_sy=2),
        dict(q_mode=None, kv_mode='bsm', kv_r=0.6, kv_sx=2, kv_sy=2),
        dict(q_mode='bsm', kv_mode=None, q_r=0.8, q_sx=4, q_sy=4),
        dict(q_mode='bsm', kv_mode=None, q_r=0.8, q_sx=4, q_sy=4)
    ],
    'bsm_fast': [
        dict(q_mode=None, kv_mode='bsm_r2D', kv_r=0.9, kv_sx=4, kv_sy=4),
        dict(q_mode=None, kv_mode='bsm_r2D', kv_r=0.9, kv_sx=4, kv_sy=4),
        dict(q_mode='bsm_r2D', kv_mode=None, q_r=0.9, q_sx=4, q_sy=4),
        dict(q_mode='bsm_r2D', kv_mode=None, q_r=0.9, q_sx=4, q_sy=4)
    ],
    'n2d_2x2': [
        dict(q_mode='neighbor_2D', kv_mode=None, q_s=(2, 2)),
        dict(q_mode='neighbor_2D', kv_mode=None, q_s=(2, 2)),
        dict(q_mode='neighbor_2D', kv_mode=None, q_s=(2, 2)),
        dict(q_mode='neighbor_2D', kv_mode=None, q_s=(2, 2))
    ]
}
