"""Explicit side-weighted BAND ablation; calibration uses the same objective."""
import ast
import inspect
import textwrap


def make_band_class(inner_weight, focal_gamma=0, steps=2, dynamic=False, adaptive=False):
    from thesis.new_constraints.bands import outer_boundary as original
    if not 0 < inner_weight < 1:
        raise ValueError('Both boundary sides must retain positive weight')
    if focal_gamma not in (0, 1, 2):
        raise ValueError('Only explicitly accounted focal exponents 0, 1, 2 are supported')
    if steps not in (1, 2):
        raise ValueError('Only one- and two-step bands are accounted for')
    if adaptive and (dynamic or inner_weight != .5 or focal_gamma != 0 or steps != 2):
        raise ValueError('Adaptive sides require the otherwise canonical BAND definition')
    if inner_weight == .5 and focal_gamma == 0 and steps == 2 and not dynamic and not adaptive:
        return original.OuterBoundaryBandLoss
    source=textwrap.dedent(inspect.getsource(original.OuterBoundaryBandLoss.forward))
    anchor='case_loss = 0.5 * (inner_loss + outer_loss)'
    if source.count(anchor)!=1:raise ValueError('Canonical BAND source changed')
    expression=('case_loss = self.hfh_inner_weight * inner_loss + (1.0 - self.hfh_inner_weight) * outer_loss'
                if dynamic else f'case_loss = {inner_weight!r} * inner_loss + {1-inner_weight!r} * outer_loss')
    if adaptive:
        expression = ('_hfh_inner = inner_loss.detach()\n'
                      '_hfh_outer = outer_loss.detach()\n'
                      '_hfh_weight = ((_hfh_inner + 1e-6) / '
                      '(_hfh_inner + _hfh_outer + 2e-6)).clamp(.1, .9)\n'
                      'case_loss = _hfh_weight * inner_loss + (1.0 - _hfh_weight) * outer_loss')
        # Preserve the canonical forward indentation for the added statements.
        line = next(line for line in source.splitlines() if anchor in line)
        indent = line[:len(line)-len(line.lstrip())]
        expression = expression.replace('\n', '\n'+indent)
    source=source.replace(anchor,expression)
    tree = ast.parse(source)
    if focal_gamma:
        class AddFocal(ast.NodeTransformer):
            count = 0

            def visit_Assign(self, node):
                if any(isinstance(t, ast.Name) and t.id == 'voxel_bce' for t in node.targets):
                    self.count += 1
                    extra = ast.parse(
                        '_hfh_probability = torch.sigmoid(log_odds)\n'
                        '_hfh_pt = torch.where(target.bool(), _hfh_probability, 1.0 - _hfh_probability)\n'
                        f'voxel_bce = (1.0 - _hfh_pt).pow({focal_gamma}) * voxel_bce\n'
                    ).body
                    return [node, *extra]
                return node

        transform = AddFocal()
        tree = transform.visit(tree)
        if transform.count != 1:
            raise ValueError('Canonical voxel BCE assignment changed')
        ast.fix_missing_locations(tree)
    namespace=dict(vars(original))
    exec(compile(tree,__file__,'exec'),namespace)
    class SideWeightedBand(original.OuterBoundaryBandLoss):
        forward=namespace['forward']
        hfh_inner_weight=inner_weight

        def __init__(self, *args, **kwargs):
            # Initialize the canonical defaults, then explicitly set the width
            # for this separately labelled ablation (canonical class stays 2).
            kwargs['steps'] = 2
            super().__init__(*args, **kwargs)
            self.steps = steps
    return SideWeightedBand


def install(protocol):
    from scripts.adni_constraints import objectives as binary
    binary.OuterBoundaryBandLoss=make_band_class(protocol.get('band_inner_weight', .5), protocol.get('band_focal_gamma', 0), protocol.get('band_steps', 2), 'band_final_inner_weight' in protocol, protocol.get('band_adaptive_sides', False))


def side_weight_for_step(initial, final, start, charged, maximum):
    if not (0 < initial < 1 and 0 < final < 1 and 0 <= start < maximum-1):
        raise ValueError('Invalid side-weight transition')
    fraction=max(0.,min(1.,(charged-start)/(maximum-1-start)))
    return initial+(final-initial)*fraction


def update_side_weight(protocol, charged, maximum):
    from scripts.adni_constraints import objectives as binary
    binary.OuterBoundaryBandLoss.hfh_inner_weight=side_weight_for_step(
        protocol['band_inner_weight'], protocol['band_final_inner_weight'],
        protocol['band_side_transition_start'], charged, maximum)
    return binary.OuterBoundaryBandLoss.hfh_inner_weight


def reset_sgd_momentum(optimizer):
    """Discard native-loss velocity at the declared BAND phase transition."""
    import torch
    if not isinstance(optimizer, torch.optim.SGD):
        raise ValueError('This transition ablation is defined only for SGD')
    entries = len(optimizer.state)
    optimizer.state.clear()
    return entries


def release_factor(charged, maximum, release_updates):
    if not 2 <= release_updates <= maximum:
        raise ValueError('Invalid terminal release length')
    start = maximum - release_updates
    if charged < start:
        return 1.0
    return max(0.0, (maximum - 1 - charged) / (release_updates - 1))


def install_training_transition(trainer):
    source = textwrap.dedent(inspect.getsource(trainer.run))
    # Prefix-zero calibration commits before the first training charge. The
    # recovery writer requires a ledger even for that zero-update snapshot.
    ledger_anchor = '    step, best_score, best_step, epoch, cursor = 0, -1., 0, 0, 0'
    if source.count(ledger_anchor) != 1:
        raise ValueError('Initial ledger hook changed')
    source = source.replace(ledger_anchor,
        '    if not ledger_path.exists():\n        base.atomic_json(ledger_path, ledger)\n' + ledger_anchor)
    anchor = '        optimizer.step()'
    if source.count(anchor) != 1:
        raise ValueError('Training update hook changed')
    replacement = '''        if protocol.get("band_reset_momentum") and charged == schedule["prefix_updates"]:
            from scripts.colab_hfh.bands_variants import reset_sgd_momentum
            entries = reset_sgd_momentum(optimizer)
            base.atomic_json(args.output / "band_optimizer_reset.json", dict(
                policy="reset_sgd_momentum_at_first_band_update", attempt=charged + 1,
                cleared_state_entries=entries))
        optimizer.step()'''
    source = source.replace(anchor, replacement)
    side_anchor = '        active_arm = "dice" if charged < schedule["prefix_updates"] else args.objective'
    if source.count(side_anchor) != 1:
        raise ValueError('Active objective hook changed')
    source = source.replace(side_anchor, '''        if charged >= schedule["prefix_updates"] and "band_final_inner_weight" in protocol:
            from scripts.colab_hfh.bands_variants import update_side_weight
            _hfh_inner_weight = update_side_weight(protocol, charged, max_steps)
            if charged == max_steps - 1:
                base.atomic_json(args.output / "band_side_transition.json", dict(
                    final_inner_weight=_hfh_inner_weight, final_update_index=charged,
                    initial_inner_weight=protocol["band_inner_weight"],
                    start_update_index=protocol["band_side_transition_start"]))
''' + side_anchor)
    lines = [line for line in source.splitlines() if line.strip().startswith('scale = .2 if active_arm')]
    if len(lines) != 1:
        raise ValueError('Constraint scale hook changed')
    source = source.replace(lines[0], lines[0] + '''
        if charged >= schedule["prefix_updates"] and protocol.get("band_release_updates"):
            from scripts.colab_hfh.bands_variants import release_factor
            scale *= release_factor(charged, max_steps, protocol["band_release_updates"])''')
    # Retain the original module globals, including signal/recovery state.
    exec(compile(source, __file__, 'exec'), vars(trainer))
