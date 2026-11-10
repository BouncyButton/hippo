"""Frozen, provenance-checked atlas cache shared by matched arms."""
import json
from pathlib import Path
import numpy as np
from scripts.depth_supervision.utils import atomic_json, digest
from .geometry import build_atlas


def field_of_view(shape, native_shape):
    before = (np.array(shape) - native_shape) // 2
    result = np.zeros(shape, bool)
    result[tuple(slice(int(a), int(a + b)) for a, b in zip(before, native_shape))] = True
    return result


class AtlasCache:
    def __init__(self, directory, protocol, *, require_complete=True):
        self.directory = Path(directory)
        self.protocol = protocol
        manifest_path = self.directory / 'manifest.json'
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if require_complete and not self.manifest.get('complete'):
            raise ValueError('The complete geometry preflight must pass before branch training')
        if require_complete and self.manifest.get('protocol_sha256') != protocol['identity_sha']:
            raise ValueError('Atlas protocol identity changed')
        self.verified = set()

    def get(self, item):
        path = self.directory / (item['case_name'] + '.npz')
        if item['case_name'] not in self.verified:
            if digest(path) != self.manifest['artifacts'][path.name]:
                raise ValueError('Frozen atlas changed: ' + str(path))
            self.verified.add(item['case_name'])
        with np.load(path, allow_pickle=False) as source:
            return {k: source[k] for k in source.files}


def prepare(patients, predictions, protocol, directory, provenance):
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    results = []
    for item in patients.items:
        name = item['case_name']
        output, report_path = directory / (name + '.npz'), directory / (name + '.json')
        prediction = predictions[name]
        image = item['image'][0].numpy()
        fov = field_of_view(image.shape, protocol['native_geometry'][name]['native_shape'])
        if not output.exists() or not report_path.exists():
            atlas, report = build_atlas(image, prediction, item['affine'], fov,
                                       margin_mm=protocol['atlas']['margin_mm'],
                                       chart_step=protocol['atlas']['chart_step'])
            # Labels enter only these diagnostics after the input construction.
            labels = item['label'][0].numpy()
            covered = np.zeros(labels.size, bool); covered[atlas['roi_indices']] = True
            covered = covered.reshape(labels.shape)
            missed = (labels > 0) & (prediction == 0)
            report['annotated_tissue_coverage'] = float(covered[labels > 0].mean())
            report['missed_tissue_coverage'] = float(covered[missed].mean()) if missed.any() else None
            report['prediction_foreground_voxels'] = int((prediction > 0).sum())
            report['case'] = name
            report['provenance'] = provenance[name]
            report['protocol_sha256'] = protocol['identity_sha']
            if any(x['return_support_fraction'] < .999 for x in report['charts_audit']):
                raise ValueError('Some source samples have no reconstruction support: ' + name)
            if any(x['flipped_triangles'] for x in report['surface']):
                raise ValueError('Surface chart flips: ' + name)
            temporary = output.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **atlas)
            temporary.replace(output)
            report['atlas_sha256'] = digest(output)
            atomic_json(report_path, report)
        report = json.loads(report_path.read_text())
        if report['provenance'] != provenance[name]:
            raise ValueError('Guide provenance changed: ' + name)
        if report['protocol_sha256'] != protocol['identity_sha'] or report['atlas_sha256'] != digest(output):
            raise ValueError('Frozen atlas identity changed: ' + name)
        results.append(report)
        print(json.dumps(dict(stage='atlas', case=name, completed=len(results), total=len(patients),
                              charts=report['charts'], tissue_coverage=report['annotated_tissue_coverage'])), flush=True)
    manifest = dict(complete=True, protocol_sha256=protocol['identity_sha'], cases=[r['case'] for r in results],
                    artifacts={p.name: digest(p) for p in directory.glob('*.npz')}, reports=results)
    atomic_json(directory / 'manifest.json', manifest)
    return manifest
