import pystk

from .utils import PyTux


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

    
    targetSpeed = 50

    if abs(aim_point[0]) >= 0.9:
      action.steer = aim_point[0]/abs(aim_point[0])
      action.drift = False
      targetSpeed = 1
      
    elif abs(aim_point[0]) >= 0.8:
      action.steer = aim_point[0]/abs(aim_point[0])
      action.drift = False
      targetSpeed = 2
      
    elif abs(aim_point[0]) >= 0.7:
      action.steer = aim_point[0]/abs(aim_point[0])
      action.drift = False
      targetSpeed = 6

    elif abs(aim_point[0]) >= 0.6:
      action.steer = aim_point[0]/abs(aim_point[0])
      action.drift = False
      targetSpeed = 10

    elif abs(aim_point[0]) >= 0.5:
      action.steer = aim_point[0]/abs(aim_point[0])
      action.drift = True
      targetSpeed = 15
      
    elif abs(aim_point[0]) >= 0.4 :
      action.steer = aim_point[0]/abs(aim_point[0])
      action.drift = True
      targetSpeed = 20

    elif abs(aim_point[0]) >= 0.3:
      action.steer = aim_point[0]/abs(aim_point[0])
      action.drift = True
      targetSpeed = 24
      
    elif abs(aim_point[0]) >= 0.2:
      action.steer = 3 * aim_point[0]
      action.drift = True
      targetSpeed = 28
    
    elif abs(aim_point[0]) >= 0.1:
      action.steer = 2.5 * aim_point[0]
      action.drift = False
      targetSpeed = 30
      
    else:
      action.steer = 2 * aim_point[0]
      action.drift = False
      targetSpeed = 35

    action.nitro = False

    if current_vel < 0.5 * targetSpeed:
      action.nitro = True
      action.acceleration = 0.6
      
    elif current_vel < 0.6 * targetSpeed:
      action.nitro = True
      action.acceleration = 0.5
      
    elif current_vel < 0.7 * targetSpeed:
      action.nitro = True
      action.acceleration = 0.4
      
    elif current_vel < 0.8 * targetSpeed:
      action.acceleration = 0.3
      
    elif current_vel < 0.9 * targetSpeed:
      action.acceleration = 0.2
      
    elif current_vel < 1 * targetSpeed:
      action.acceleration = 0.1
      
    elif current_vel < 1.1 * targetSpeed:
      action.acceleration = 0.05
      action.brake = True
    elif current_vel < 1.2 * targetSpeed:
      action.acceleration = 0.01
      action.brake = True
    elif current_vel < 1.3 * targetSpeed:
      action.acceleration = 0.01
      action.brake = True
    elif current_vel < 1.4 * targetSpeed:
      action.acceleration = 0.01
      action.brake = True
    elif current_vel < 1.5 * targetSpeed:
      action.acceleration = 0.01
      action.brake = True
    else:
      action.acceleration = 0.01
      action.brake = True
      
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
