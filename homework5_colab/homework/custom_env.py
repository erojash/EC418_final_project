from gym import Env
from gym import spaces
import random
import numpy as np
import pystk
import matplotlib.pyplot as plt

import utils
import collections
import os

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
MAX_FRAMES = 1300
ON_COLAB = True

class kartEnv(Env):
    def __init__(self):

        # self.observation_space = spaces.Dict(
        #     {
        #         "aimPoint" : spaces.Box(low=-1, high= 1, shape=(1,))
        #         "speed" : spaces.Box(low=0, high=50, shape=(1,)),
        #     }
        # )
            
        # self.action_space = spaces.Dict(
        #     {
        #         "steer" : spaces.Box(low=-1, high= 1, shape=(1,)),
        #         "acceleration" : spaces.Box(low=0, high= 1, shape=(1,)),
        #         "drift" : spaces. Discrete(2,start=0),
        #         "nitro" : spaces. Discrete(2,start=0),
        #         "brake" : spaces. Discrete(2,start=0)
        #     }
        # )
        
        self.observation_space = spaces.Box(np.array([-1,-10]), np.array([1,30]))
        
        self.action_space = spaces.Box(np.array([-1,0.001,0,0,0]), np.array([1,1,0.0002,0.0002,0.0002]))
        
        # initialize current state 
        self.currentFrame = 0
        
        if ON_COLAB:
            self.images = list()
        
        # initialize the pytux
        self.pytux = 0
        if isinstance(self.pytux, utils.PyTux):
            self.reset()
        else:
            self.pytux = utils.PyTux()

        self.trackName = 'lighthouse'
        
        self.reset()
        
    def step(self, action):
        
        info = {}
        
        kart = self.state.players[0].kart
        
        if np.isclose(kart.overall_distance / self.track.length, 1.0, atol=2e-3):  
            done = True
            reward = 20000
            
        elif self.currentFrame  >= MAX_FRAMES -1:
            done = True   
            reward = -20000
        else:
            done = False
            # reward = -5 + (3 * kart.overall_distance / self.track.length)**2
            reward = -1 * (4 * self.currentFrame / MAX_FRAMES) **2
            reward += (4 * kart.overall_distance / self.track.length)**2
            if kart.overall_distance / self.track.length > 0.9 and 75 < self.currentFrame < 200:
                reward += - 10000
           
        self.terminate = done
            
        self.currentFrame += 1
        
        # if planner:
        #         image = np.array(self.k.render_data[0].image)
        #         aim_point = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
        
        kartAction = pystk.Action()
        kartAction.steer = action[0]
        kartAction.acceleration = action[1]
        if action[2] > 0.0001:
            kartAction.drift = True
        else:
            kartAction.drift = False
            
        if action[3] > 0.0001:
            kartAction.brake = True
        else:
            kartAction.brake = False
        if action[4] > 0.0001:
            kartAction.nitro = True
        else:
            kartAction.nitro = False
        
        
        current_vel = np.linalg.norm(kart.velocity)
        
        t = self.currentFrame
        if current_vel < 1.0 and t - self.last_rescue > RESCUE_TIMEOUT:
            self.last_rescue = t
            kartAction.rescue = True
            reward -= 5000
                
        self.pytux.k.step(kartAction)
        
        # get the observation
        self.state.update()
        self.track.update()

        kart = self.state.players[0].kart

        proj = np.array(self.state.players[0].camera.projection).T
        view = np.array(self.state.players[0].camera.view).T

        aim_point_world = self.pytux._point_on_track(kart.distance_down_track+TRACK_OFFSET, self.track)
        aim_point_image = self.pytux._to_image(aim_point_world, proj, view)
        aim_point = aim_point_image
        current_vel = np.linalg.norm(kart.velocity)
            
            
        # observation = {
        #     "speed" : [current_vel],
        #     "aimPoint" : [aim_point[0]]
        # }
        # observation = collections.OrderedDict(observation)
        
        if current_vel >30:
            current_vel = 30
        elif current_vel < -10:
            current_vel = -10
        observation = np.array([aim_point[0], current_vel])
        return observation, reward, done, info
    
    def render(self):
        
        kart = self.state.players[0].kart
        
        proj = np.array(self.state.players[0].camera.projection).T
        view = np.array(self.state.players[0].camera.view).T
        
        aim_point_world = self.pytux._point_on_track(kart.distance_down_track+TRACK_OFFSET, self.track)
        aim_point_image = self.pytux._to_image(aim_point_world, proj, view)
        aim_point = aim_point_image
        
        # if not ON_COLAB:
        #     self.fig, self.ax = plt.subplots(1, 1)
        #     self.ax.clear()
        #     self.ax.imshow(self.pytux.k.render_data[0].image)
        #     WH2 = np.array([self.pytux.config.screen_width, self.pytux.config.screen_height]) / 2
        #     self.ax.add_artist(plt.Circle(WH2*(1+self.pytux._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
        #     self.ax.add_artist(plt.Circle(WH2*(1+self.pytux._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
        #     # if planner:
        #     #     ap = self.pytux._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
        #     #     ax.add_artist(plt.Circle(WH2*(1+aim_point_image), 2, ec='g', fill=False, lw=1.5))
        #     plt.pause(1e-3)
        # elif ON_COLAB:
        if ON_COLAB:
            from PIL import Image, ImageDraw
            image = Image.fromarray(self.pytux.k.render_data[0].image)
            draw = ImageDraw.Draw(image)

            WH2 = np.array([self.pytux.config.screen_width, self.pytux.config.screen_height]) / 2

            p = (aim_point_image + 1) * WH2
            draw.ellipse((p[0] - 2, p[1] - 2, p[0]+2, p[1]+2), fill=(255, 0, 0))
            # if planner:
            #     p = (aim_point + 1) * WH2
            #     draw.ellipse((p[0] - 2, p[1] - 2, p[0]+2, p[1]+2), fill=(0, 255, 0))

            self.images.append(np.array(image))
            # print("Frame: ", self.currentFrame, self.terminate, flush=True)
    
    def playVideo(self):
        print("finished at frame: ", self.currentFrame)
        
        if ON_COLAB:
            from moviepy.editor import ImageSequenceClip
            from IPython.display import display

            display(ImageSequenceClip(self.images, fps=15).ipython_display(width=512, autoplay=True, loop=True, maxduration=120))
    
    def changeTrack(self, track):
        self.trackName = track
        self.reset()
    
    def reset(self):
        
        
        if self.pytux.k is not None and self.pytux.k.config.track == self.trackName:
            self.pytux.k.restart()
            self.pytux.k.step()
        else:
            if self.pytux.k is not None:
                self.pytux.k.stop()
                del self.pytux.k
            config = pystk.RaceConfig(num_kart=1, laps=1, track=self.trackName)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

            self.pytux.k = pystk.Race(config)
            self.pytux.k.start()
            self.pytux.k.step()
            
        self.state = pystk.WorldState()
        self.track = pystk.Track()

        self.last_rescue = 0
        self.state.update()
        self.track.update()
        
        kart = self.state.players[0].kart

        proj = np.array(self.state.players[0].camera.projection).T
        view = np.array(self.state.players[0].camera.view).T

        aim_point_world = self.pytux._point_on_track(kart.distance_down_track+TRACK_OFFSET, self.track)
        aim_point_image = self.pytux._to_image(aim_point_world, proj, view)
        aim_point = aim_point_image
        current_vel = np.linalg.norm(kart.velocity)
        
        self.currentFrame = 0
        self.terminate = False
        
        if ON_COLAB:
            self.images = list()
            
        # observation = {
        #     "speed" : [current_vel],
        #     "aimPoint" : [aim_point[0]]
        # }
        # observation = collections.OrderedDict(observation)
        
        if current_vel >30:
            current_vel = 30
        elif current_vel < -10:
            current_vel = -10
        observation = np.array([aim_point[0], current_vel])
        
        return observation
        
    def close(self):
        if isinstance(self.pytux, utils.PyTux()):
            self.pytux.close()




