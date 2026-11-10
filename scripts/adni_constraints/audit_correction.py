"""Charge binary grouped log-odds reductions absent from the old 3-class audit."""


def correct_row(row):
    for key, operator in row['operator_audit'].items():
        if key.endswith(':aten.logsumexp.default') and operator['supplementary_category'] == 'no_supplement':
            # Stable logsumexp: subtract/exp for N inputs; N-G summation;
            # G logarithms and G additions. Max comparisons are excluded.
            cost = 3 * operator['input_float_elements'] + operator['output_float_elements']
            stage = key.split(':', 1)[0]
            operator.update(supplementary_flops=cost, supplementary_category='binary_logsumexp')
            row['supplementary_per_stage'][stage] = row['supplementary_per_stage'].get(stage, 0) + cost
            row['flops'] += cost
    return row


def correct_profile(profile):
    for row in profile['profiles']:
        correct_row(row)
    for arm in profile['costs']:
        rows = [r for r in profile['profiles'] if r['arm'] == arm]
        train = {r['flops'] for r in rows if not r['calibration']}
        cal = {r['flops'] for r in rows if r['calibration']}
        if len(train) != 1 or (arm != 'dice' and len(cal) != 1):
            raise ValueError('Corrected arithmetic depends on geometry')
        profile['costs'][arm] = dict(update_flops=train.pop(),
            calibration_flops=10*cal.pop()+(4096 if 'depth' in arm else 1024) if cal else 0)
    profile['binary_audit_extension'] = 'Stable logsumexp charged as 3*N+G; one logit per binary group. Idempotently reconciled with recorded operator counts.'
    return profile
