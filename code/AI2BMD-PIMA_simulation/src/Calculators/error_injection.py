import numpy as np

#                                   <--- on --->                 <if periodic>
#  ERR_ON                           ____________                 __________
#                                  |            |               |
#                                  |            |               |
#                                  |            |               |
#                                  |            |               |
#                                  |            |               |
#  ERR_OFF ________________________|            |_____ ... ____<:__________
#           <---grace---><---off--->                             <if non-periodic>

grace = 0
off = -1
on = -1
periodic = False

_cur_step = 0


def initialize(args: str):
    global grace, off, on, periodic, _cur_step
    _cur_step = 0
    if not args:
        grace, off, on, periodic = 0, -1, -1, False
    else:
        grace, off, on, periodic = map(int, args.split(','))

def should_inject():
    global _cur_step, grace

    if grace > 0:
        grace -= 1
        return False

    T = on + off
    if _cur_step >= T and periodic:
        _cur_step = 0

    ERR_ON = off <= _cur_step < T
    _cur_step += 1
    return ERR_ON


# note: as soon as I start to inject error at the scale of max_force * 5,
# it breaks within the next 5 steps.
# problem: the bondlen check is always behind the pace.
# it will not detect the initial erronous forces, but rather, ONE STEP AFTER THAT
def inject_force_errors(forces: np.ndarray):

    max_force = np.max(forces)
    # print(f"force max value: {max_force}")
    forces += np.random.randn(*forces.shape) * max_force * 5
