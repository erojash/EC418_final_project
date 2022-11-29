import torch
import torch.nn.functional as F
import torch.nn as nn # RM


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    def __init__(self):

      super().__init__()
    
      layers = []
      layers.append(torch.nn.Conv2d(3,16,5,2,2))
      layers.append(torch.nn.BatchNorm2d(16))
      layers.append(torch.nn.ReLU())

      layers.append(torch.nn.Conv2d(16,16,5,2,2))
      layers.append(torch.nn.BatchNorm2d(16))
      layers.append(torch.nn.ReLU())
     
      layers.append(torch.nn.AvgPool2d(2, 2))

      layers.append(torch.nn.Conv2d(16, 32, 5, 2, 2))
      layers.append(torch.nn.BatchNorm2d(32))
      layers.append(torch.nn.ReLU())

      layers.append(torch.nn.Conv2d(32, 32, 5, 2, 2))
      layers.append(torch.nn.BatchNorm2d(32))
      layers.append(torch.nn.ReLU())

      
      self._conv = torch.nn.Sequential(*layers)



    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        x = self._conv(img)
        
        
        return spatial_argmax(x[:, 0])
        #return self.classifier(x.mean(dim=[-2, -1]))

    def num_flat_features(self, x):
      size = x.size()[1:]  # all dimensions except the batch dimension
      num_features = 1
      for s in size:
          num_features *= s
      return num_features


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


def test_planner(pytux, track, verbose=False):
    from .controller import control

    track = [track] if isinstance(track, str) else track
    planner = load_model().eval()

    for t in track:
        steps, how_far = pytux.rollout(t, control, planner, max_frames=1000, verbose=verbose)
        print(steps, how_far)

if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')

    pytux = PyTux()
    test_planner(pytux, **vars(parser.parse_args()))
    pytux.close()
