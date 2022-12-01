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
        
        # input: B,3,96,128
        layers.append(torch.nn.MaxPool2d(2, 2)) 
        
        # input: B,3,48,64
        # channels:3 -> 8, kernel: 4x4, stride: 2x2, padding: 2x2
        layers.append(torch.nn.Conv2d(3,16,4,2,2))
        layers.append(torch.nn.LeakyReLU())
        
        # input: B,16,25,33
        layers.append(torch.nn.Conv2d(16,16,4,1,2))
        layers.append(torch.nn.LeakyReLU())
        
        # input: B,16,26,34
        layers.append(torch.nn.Flatten())
        
        # input: B,16*26*34
        layers.append(torch.nn.Linear(16*26*34, 1024))
        layers.append(torch.nn.LeakyReLU())
        layers.append(torch.nn.Linear(1024, 256))
        layers.append(torch.nn.LeakyReLU())
        layers.append(torch.nn.Linear(256, 32))
        layers.append(torch.nn.LeakyReLU())
        layers.append(torch.nn.Linear(32, 2))

        self._conv = torch.nn.Sequential(*layers)



    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        x = self._conv(img)
        # print(img.shape, x.shape)
        # print(img.shape)
        # print(x.shape)
        
        return x
        # return spatial_argmax(x[:, 0])
        # return self.classifier(x.mean(dim=[-2, -1]))


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
