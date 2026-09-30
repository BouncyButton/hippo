import numpy as np
import pytest

from .audit_sdf_shoulder import signed_distance,signed_curvatures,envelope_depth,shapes_for_mask


def mask_with_notch():
    mask=np.zeros((13,45,36),dtype=bool)
    for y in range(5,42):
        top=int(np.rint(18+.025*(y-23)**2))
        mask[3:10,y,4:top+1]=True
    return mask


def test_signed_distance_has_correct_sign():
    mask=np.zeros((15,15,15),dtype=bool);mask[4:11,4:11,4:11]=True
    phi=signed_distance(mask)
    assert np.all(phi[mask]<0) and np.all(phi[~mask]>0)


def test_curvature_sign_for_analytic_upper_concavity_and_convexity():
    x,y,z=np.meshgrid(np.arange(9),np.arange(41),np.arange(35),indexing='ij')
    concave_phi=z-(14+.025*(y-20)**2)
    convex_phi=z-(26-.025*(y-20)**2)
    a,h=signed_curvatures(concave_phi)
    b,_=signed_curvatures(convex_phi)
    assert a[4,20,14] == pytest.approx(-.05)
    assert h[4,20,14]<0
    assert b[4,20,26] == pytest.approx(.05)


def test_upper_envelope_depth_detects_notch_not_convex_cap():
    y=np.arange(21,dtype=float)
    inward=10+.1*(y-10)**2
    depth=envelope_depth(y,inward)
    assert np.argmax(depth)==10
    assert depth[10]==pytest.approx(10)
    assert np.max(envelope_depth(y,30-inward))==pytest.approx(0)
    np.testing.assert_allclose(envelope_depth(y,20-.3*y),0,atol=1e-12)


def test_extrema_sdf_combination_selects_synthetic_notch():
    families,d=shapes_for_mask(mask_with_notch())
    for family in ('extrema_depth','sdf_extrema'):
        s=families[family]
        peak=s.candidates[np.argmax(s.consensus)]
        # Rasterisation makes a flat valley: depth peaks at its centre, while
        # curvature peaks at either shoulder of that flat minimum.
        tolerance=1 if family=='extrema_depth' else 4
        assert abs(peak-23.5)<=tolerance
    assert d['upper_inward_fraction']>0
    assert d['occupied_columns']>0


def test_padding_native_coordinates_and_left_right_mirror():
    mask=mask_with_notch()
    a,_=shapes_for_mask(mask)
    b,_=shapes_for_mask(np.pad(mask,((2,4),(7,8),(3,5))),origin_y=-7)
    c,_=shapes_for_mask(mask[::-1])
    for family in a:
        np.testing.assert_array_equal(a[family].candidates,b[family].candidates)
        np.testing.assert_allclose(a[family].consensus,b[family].consensus,atol=1e-10)
        np.testing.assert_allclose(a[family].consensus,c[family].consensus,atol=1e-10)


def test_envelope_rejects_unordered_points():
    with pytest.raises(ValueError):envelope_depth(np.array([2,1]),np.array([0,0]))
