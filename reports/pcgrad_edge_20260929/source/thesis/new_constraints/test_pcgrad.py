import pytest
import torch

from thesis.new_constraints.pcgrad import project, backward_tasks


def explicit_reference(gradients, orders):
    projected = gradients.clone()
    for i, order in enumerate(orders):
        for j in order:
            dot, norm2 = projected[i] @ gradients[j], gradients[j].square().sum()
            if dot < 0 and norm2 > 0:
                projected[i] -= dot/norm2 * gradients[j]
    return projected.sum(0), projected


def test_projection_matches_independent_full_vector_algorithm():
    gen = torch.Generator().manual_seed(21)
    g = torch.randn(5, 57, generator=gen, dtype=torch.double)
    g[1] = -2*g[0] + .2*g[1]
    combined, report = project(g, torch.Generator().manual_seed(7))
    expected, rows = explicit_reference(g, report['orders'])
    torch.testing.assert_close(combined, expected)
    torch.testing.assert_close(torch.tensor(report['projected_task_gram']), (rows @ rows.T).float())
    assert report['projection_count'] > 0


def test_nonconflicting_zero_and_opposing_tasks():
    for g in (torch.eye(5), torch.zeros(5,7), torch.ones(5,7)):
        got, report = project(g, torch.Generator().manual_seed(2))
        torch.testing.assert_close(got,g.sum(0))
        assert report['projection_count'] == 0
    g = torch.tensor([[1.,0.],[-1.,1.]])
    got,_ = project(g,torch.Generator().manual_seed(2))
    torch.testing.assert_close(got,torch.tensor([.5,1.5]))
    got,_ = project(torch.tensor([[1.,0.],[-2.,0.],[0.,0.]]),torch.Generator().manual_seed(2))
    torch.testing.assert_close(got,torch.zeros(2))


def test_sum_control_and_dedicated_rng_replay():
    torch.manual_seed(2)
    g=torch.randn(5,11)
    model_rng=torch.get_rng_state().clone()
    rng=torch.Generator().manual_seed(8)
    state=rng.get_state()
    first,report=project(g,rng)
    rng.set_state(state)
    second,repeat=project(g,rng)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert report==repeat and torch.equal(torch.get_rng_state(),model_rng)
    total,report=project(g,rng,enabled=False)
    torch.testing.assert_close(total,g.sum(0),rtol=0,atol=0)
    assert report['projection_count']==0


def test_backward_sum_matches_joint_loss_and_unused_parameters():
    x=torch.nn.Parameter(torch.tensor([.5,-1.,2.]))
    unused=torch.nn.Parameter(torch.tensor([9.]))
    tasks=[x.square().sum(),(x-1).square().sum(),(2*x+1).square().sum()]
    expected=torch.autograd.grad(sum(tasks),x,retain_graph=True)[0]
    scaler=torch.cuda.amp.GradScaler(enabled=False)
    report=backward_tasks(tasks,(x,unused),scaler,torch.Generator().manual_seed(5),project_conflicts=False)
    torch.testing.assert_close(x.grad,expected)
    assert unused.grad is None and report['finite']


def test_nonfinite_gradients_are_not_silently_projected():
    with pytest.raises(ValueError):
        project(torch.tensor([[float('nan'),0.]]),torch.Generator())
    x=torch.nn.Parameter(torch.ones(2))
    with pytest.raises(FloatingPointError):
        backward_tasks([x.sum()*float('inf')],(x,),torch.cuda.amp.GradScaler(enabled=False),
                       torch.Generator(),project_conflicts=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA AMP check runs inside allocated GPU preflight')
def test_amp_projection_unscaling_and_overflow_skip():
    gradients=torch.randn(5,127,device='cuda')
    gradients[2]=-gradients[0]*.001
    combined,report=project(gradients,torch.Generator().manual_seed(3))
    expected,_=explicit_reference(gradients.double().cpu(),report['orders'])
    torch.testing.assert_close(combined,expected.cuda().float(),rtol=1e-5,atol=1e-6)
    model=torch.nn.Linear(4,3).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    scaler=torch.cuda.amp.GradScaler(init_scale=128.)
    x=torch.randn(5,4,device='cuda')
    with torch.autocast('cuda'):
        y=model(x)
        tasks=[y.float().square().mean(),(y.float()-1).square().mean()]
    expected=torch.autograd.grad(scaler.scale(sum(tasks)),tuple(model.parameters()),retain_graph=True)
    backward_tasks(tasks,tuple(model.parameters()),scaler,torch.Generator().manual_seed(4),project_conflicts=False)
    for p,g in zip(model.parameters(),expected):
        torch.testing.assert_close(p.grad,g,rtol=.01,atol=.01)
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
    before=[p.detach().clone() for p in model.parameters()]
    scale_before=scaler.get_scale()
    loss=model(x).sum()*float('inf')
    report=backward_tasks([loss],tuple(model.parameters()),scaler,torch.Generator(),project_conflicts=True)
    scaler.step(optimizer);scaler.update()
    assert not report['finite'] and scaler.get_scale()<scale_before
    for p,v in zip(model.parameters(),before):
        torch.testing.assert_close(p,v,rtol=0,atol=0)
