"""Correspondence and physical-geometry tests for the volume reformat."""
import unittest

import numpy as np

from .geometry import Straightening, fit_straightening, matched_axes, inside_grid, sample_grid


class GeometryTests(unittest.TestCase):
    def setUp(self):
        s = np.linspace(-15., 15., 31)
        self.transform = Straightening(np.array([8.,2.,-4.]), np.eye(3), s,
                                       4*np.sin(s/8), s, 4*np.sin(s/8), 1.)

    def test_inverse_and_signed_depth(self):
        xyz = np.random.default_rng(0).uniform(-40, 40, (1000,3))
        q = self.transform.world_to_straight(xyz)
        np.testing.assert_allclose(self.transform.straight_to_world(q), xyz, atol=1e-12)
        np.testing.assert_allclose(q[:,2], (xyz-self.transform.center) @ self.transform.basis[:,2], atol=1e-12)

    def test_numerical_jacobian_no_foldover(self):
        delta = 1e-4
        for s in np.linspace(-25., 25., 31):
            p = np.array([s,2.,-4.])
            jac = np.stack([(self.transform.straight_to_world(p+np.eye(3)[i]*delta)-
                             self.transform.straight_to_world(p-np.eye(3)[i]*delta))/(2*delta) for i in range(3)],1)
            self.assertAlmostEqual(float(np.linalg.det(jac)),1.,places=8)

    def test_axes_cover_same_roi_in_both_views(self):
        xyz=np.random.default_rng(1).uniform(-20,20,(1000,3))
        axes=matched_axes(self.transform,xyz)
        self.assertTrue(inside_grid(axes,self.transform.world_to_local(xyz),1).all())
        self.assertTrue(inside_grid(axes,self.transform.world_to_straight(xyz),1).all())

    def test_linear_grid_resampling(self):
        axes=[np.arange(16.) for _ in range(3)]
        q=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
        field=2*q[...,0]-3*q[...,1]+q[...,2]
        query=np.random.default_rng(2).uniform(1,14,(200,3))
        np.testing.assert_allclose(sample_grid(field,axes,query),2*query[:,0]-3*query[:,1]+query[:,2],atol=1e-12)

    def test_fitting_ignores_anterior_posterior_labels(self):
        i,j,k=np.indices((32,48,32))
        mask=((i-16-4*np.sin((j-24)/12))/5)**2+((j-24)/19)**2+((k-16)/5)**2 < 1
        a=np.where(mask,np.where(j<24,1,2),0)
        b=np.where(mask,np.where(j<24,2,1),0)
        x,y=fit_straightening(a,np.eye(4)),fit_straightening(b,np.eye(4))
        np.testing.assert_array_equal(x.offsets,y.offsets)
        np.testing.assert_array_equal(x.basis,y.basis)


if __name__ == '__main__':
    unittest.main()
