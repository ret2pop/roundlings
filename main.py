import tensorflow as tf
import numpy as np
import time

import math
import random
import sys
import gc # for gc.collect()

def sig(x):
    return 1 / (1 + np.exp(-x))

class RoundlingModel():
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def compute(self):
        pass

class Board:
    def __init__(self, max_x, max_y, max_food):
        self.max_x = max_x
        self.max_y = max_y
        self.max_food = max_food
        self.time = 0
        self.roundlings = [] # list of all objects on the board
        self.proteins = []
        self.food = []

    def populate(self, roundlings_count):
        for i in range(roundlings_count): # TODO Figure out how tf.random.uniform works
            rand_init = tf.random.uniform(shape=[3], minval=0., maxval=5., seed=22)
            self.roundlings.append(Roundling())
        for i in range(self.max_food):
            self.food.append(Food(random.random() * self.max_x, random.random() * self.max_y, random.random() * 10 + 5, self))

    def is_outside(self, iobj):
        return iobj.x < 0 or iobj.y < 0 or iobj.x > self.max_x or iobj.y > self.max_y

    def step_time(self):
        delta = 0.1

        for p in proteins:
            p.turn()
            p.time += delta
        self.proteins[:] = [p for p in self.protiens if not p.time >= 5 or self.is_outside(p)]
        gc.collect()

        for f in food:
            f.turn()
            proteins.append(f.make_protein())
        self.food[:] = [f for f in self.food if not self.is_outside(f)]
        gc.collect()

        for r in self.roundlings:
            r.turn()
            r.time += delta
            for p in protein:
                if r.is_interacting(p) and r != p.creator_obj:
                    r.proteins.append(p)

        self.time += delta

        for r in self.roundlings:
            for f in food:
                if r.is_interacting(f) and not f.dead:
                    r.energy += f.energy
                    f.dead = True
            if r.energy > 15:
                self.roundlings.append(r.reproduce()) # TODO

        self.food[:] = [f for f in self.food if not f.dead]
        self.proteins[:] = [p for p in self.proteins if not p.dead]
        self.roundlings[:] = [r for r in self.roundlings if not r.energy <= 0 or r.time > 30 or self.is_outside(r)]
        gc.collect()

        while len(self.food) < self.max_food:
            self.food.append(Food(random.random() * self.max_x, random.random() * self.max_y, self))

class IntObject:
    def __init__(self, energy, x, y, radius, board):
       self.energy = energy
       self.x = x
       self.y = y
       self.radius = radius
       self.time = 0
       self.dead = False

    def is_interacting(self, obj):
        return self.radius + obj.radius >= math.sqrt(math.pow(self.x - obj.x, 2) + math.pow(self.y - obj.y, 2))

    def brownian_move(self):
        angle = random.random() * 2 * M_PI
        delta = 0.2
        self.x += delta * math.cos(angle)
        self.y += delta * math.sin(angle)

    def turn(self): # Take a single turn of small timestep
        self.brownian_move()


class Food(IntObject):
    def __init__(self, x, y, board):
        super().__init__(self, 1, x, y, 1, board)

    def make_protein(self):
        return Protein((random.random() * 10 - 5) + self.x, (random.random() * 10 - 5) + self.y, self, 0.1) # 0.1 is a special type


class Protein(IntObject):
    def __init__(self, x, y, creator_obj, p_type):
        super().__init__(0, x, y, 2)
        self.p_type = p_type # float that describes what type of protien it is
        self.creator_obj = creator_obj # We don't want this protein to interact with the creator


class Roundling(IntObject):
    def __init__(self, energy, x, y, radius, model, board):
        super().__init__(energy, x, y, radius)
        self.model = model
        self.board = board
        self.proteins = []

    def decide_move(self): # 2 layers TODO figure out how tf works
        pass

    def move(self, output_vector): # TODO remember to subtract energy according to an energy loss function per vector magnitude
        self.brownian_move()


    def turn(self): # Takes a turn with timespan of single timestep; remember to subtract energy proportional to move, size, protein
        self.move(self.decide_move())

    def reproduce(self): # add random vector to weights and biases
        energy = energy / 2
        child_energy = self.energy
        child_x = (random.random() * 2 - 1) + self.x
        child_y = (random.random() * 2 - 1) + self.y
        child_radius_delta = (random.random() * 2 - 1)
        while child_radius_delta + self.radius <= 0:
            child_radius_delta = (random.random() * 2 - 1)

        child_radius = (random.random() * 2 - 1) + self.radius
        # TODO: actually make the object


if __name__ == "__main__":
    random.seed(2031239)
    ROUNDLING_COUNT = 50
    FOOD_COUNT = 20

    # initialize roundlings
    board = Board(100, 100, FOOD_COUNT)
    board.populate(ROUNDLING_COUNT)

    while True:
        board.step_time()
        print(f"Time elapsed: {board.time}")
        print(f"Roundling Count: {len(board.roundlings)}")
        print(f"Protein Count: {len(board.proteins)}")
        sleep(0.5)
