"""Exact native no-augmentation replication with observational loss logging."""
import inspect

from scripts.colab_hfh import medsam_augmentation_components as logging
from scripts.colab_hfh.medsam_intensity_augmentation import training_source, augmentation_seed


def install(engine, protocol, output):
    if protocol['augmentation'] != 'none' or (protocol['fold'],protocol['seed'],protocol['updates']) != (0,0,125):
        raise ValueError('Only original no-augmentation f0s0 native125 replication is allowed')
    logging.install_prediction_logging(engine)
    reference = engine.forward_slab

    def logged_forward(model, criterion, slabs, case, anchor, update, micro):
        recorded = logging.LoggedCriterion(criterion)
        return logging.attach_terms(reference(model, recorded, slabs, case, anchor), recorded)

    engine.__dict__['hfh_augmented_forward'] = logged_forward
    exec(compile(training_source(inspect.getsource(engine.train)), __file__, 'exec'), engine.__dict__)
    tape = []
    for index,(case,anchor) in enumerate(protocol['frozen_sample_schedule']):
        update,micro = divmod(index,protocol['accumulation'])
        tape.append(dict(update=update+1,micro=micro,case=case,anchor=anchor,
            seed=augmentation_seed(protocol,update,micro,case,anchor),component='none',gamma=1.,gain=1.,noise_std=0.))
    engine.write(output/'augmentation_schedule.json',tape)


install_calibration = logging.install_calibration
