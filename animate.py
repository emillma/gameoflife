# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:57:23 2020

@author: emilm
"""


from matplotlib import pyplot as plt
from matplotlib import animation, cm
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.animation import FuncAnimation
import numpy as np

normalizor = Normalize(0, 1, True)


class InteractiveGameOfLife:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.current_gen = 0
        self.fig, self.ax  = plt.subplots(1,1)
        self.image = self.ax.imshow(self.data[self.current_gen],
                                    aspect='equal',
                                    interpolation='none',
                                    cmap = cm.gray,
                                    norm=normalizor)
        self.generation_text = self.ax.text(0, 1.01, f'0/{self.data.shape[0]}',
                                            transform=self.ax.transAxes)
        self.fig.canvas.mpl_connect('button_press_event',
                                    self.button_press_event)

        self.fig.canvas.mpl_connect('button_release_event',
                                    self.button_release_event)

        self.fig.canvas.mpl_connect('key_press_event',
                                    self.key_press_event)

        self.fig.canvas.mpl_connect('motion_notify_event',
                                    self.motion_notify_event)


    def redraw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def button_press_event(self, event):
        pass

    def button_release_event(self, event):
        pass

    def key_press_event(self, event):
        if event.key == 'n':
            self.current_gen = (self.current_gen + 1) % self.shape[0]
        elif event.key == 'b':
            self.current_gen = (self.current_gen - 1) % self.shape[0]
        self.generation_text.set_text(
            f'{self.current_gen}/{self.data.shape[0]}')
        self.image.set_data(self.data[self.current_gen])
        self.redraw()
    def motion_notify_event(self, event):
        pass


def to_GIF(data):
    fig, ax  = plt.subplots(1,1)
    n = data.shape[0]
    generation_text = ax.text(0, 1.01, f'0/{n}',
                                            transform=ax.transAxes)
    normalizor = Normalize(0, 1, True)
    image = ax.imshow(data[0] * 0,
                      aspect='equal',
                      interpolation='none',
                      cmap = cm.gray,
                      norm=normalizor)
    def update(i):
        generation_text.set_text(
            f'{i}/{n}')
        image.set_data(data[i])
        return generation_text, image
    anim = FuncAnimation(fig, update, frames=range(n), interval=50)
    anim.save('line.mp4', dpi=240)
    return anim

