import pystk

from .utils import PyTux

import math
import numpy as np

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    x = aim_point[0]
    y = aim_point[1]
    y = -y

    angle = np.arctan(x/y)

    if (current_vel < 20):
      action.acceleration  = 1
    else:
      action.acceleration = 0
    
    if x <= 0.2 and x >= -0.2:
      action.steer = 0
    elif (x >= -1 and x <= -0.7):
      action.steer = -1
      action.acceleration = 0
    elif (x <= 1 and x >= 0.7):
      action.steer = 1
      action.acceleration = 0
    else:
      action.steer = angle

    # sharp turns
    if (x >= -1 and x <= -0.9):
      action.drift = True
      
    elif (x >= 0.9 and x <= 1):
      action.drift = True
    
    if (x <= 0.5 and x >= -0.5):
      action.nitro = True
    
    #print(aim_point, " ", action.steer, "\n")
    return action
  

def test_controller(pytux, track, verbose=False):
    import numpy as np

    track = [track] if isinstance(track, str) else track

    for t in track:
        steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=verbose)
        print(steps, how_far)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')

    pytux = PyTux()
    test_controller(pytux, **vars(parser.parse_args()))
    pytux.close()
