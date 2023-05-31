import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


import numpy as np

import math
import random
import gc # for gc.collect()
import time

def sig(x):
    return 1 / (1 + np.exp(-x))


def copy_and_randomize_weights(original_model, new_model, noise_std_dev=0.1):
    for original_layer, new_layer in zip(original_model.layers, new_model.layers):
        if isinstance(original_layer, tf.keras.layers.Dense):  # if you want to randomize only Dense layers
            weights, biases = original_layer.get_weights()

            noise_weights = np.random.normal(0, noise_std_dev, weights.shape)
            noise_biases = np.random.normal(0, noise_std_dev, biases.shape)

            new_weights = weights + noise_weights
            new_biases = biases + noise_biases

            new_layer.set_weights([new_weights, new_biases])


class Board:
    def __init__(self, max_x, max_y, max_food):
        self.max_x = max_x
        self.max_y = max_y
        self.max_food = max_food
        self.time = 0
        self.roundlings = []  # list of all objects on the board
        self.proteins = []
        self.food = []

    def populate(self, roundlings_count):
        for i in range(roundlings_count):  # TODO Figure out how tf.random.uniform works
            model = Sequential()
            model.add(Dense(16, activation='linear', input_shape=[13]))
            model.add(Dense(16, activation='linear'))
            model.add(Dense(5, activation='linear'))
            model.compile(optimizer='adam', loss='mse')
            self.roundlings.append(Roundling(7, random.random() * self.max_x, random.random() * self.max_y, random.random() * 5, model))
        for i in range(self.max_food):
            self.food.append(Food(random.random() * self.max_x, random.random() * self.max_y))

    def is_outside(self, iobj):
        return iobj.x < 0 or iobj.y < 0 or iobj.x > self.max_x or iobj.y > self.max_y

    def step_time(self):
        delta = 0.1

        for p in self.proteins:
            p.turn()
            p.time += delta
        self.proteins[:] = [p for p in self.proteins if not p.time >= 1 or self.is_outside(p)]
        gc.collect()

        for f in self.food:
            f.turn()
            self.proteins.append(f.make_protein())
        self.food[:] = [f for f in self.food if not self.is_outside(f)]
        gc.collect()

        for r in self.roundlings:
            a = r.turn()
            if a is not None:
                self.proteins.append(a)
            r.time += delta
            for p in self.proteins:
                if r.is_interacting(p) and r != p.creator_obj:
                    r.proteins.append(p)

        self.time += delta

        for r in self.roundlings:
            for f in self.food:
                if r.is_interacting(f) and not f.dead:
                    r.energy += f.energy
                    f.dead = True
            if r.energy >= 10:
                self.roundlings.append(r.reproduce())  # TODO

        self.food[:] = [f for f in self.food if not f.dead]
        self.proteins[:] = [p for p in self.proteins if not p.dead]
        self.roundlings[:] = [r for r in self.roundlings if not r.energy <= 0 or r.time > 30 or self.is_outside(r)]
        gc.collect()

        while len(self.food) < self.max_food:
            self.food.append(Food(random.random() * self.max_x, random.random() * self.max_y))

class IntObject:
    def __init__(self, energy, x, y, radius):
       self.energy = energy
       self.x = x
       self.y = y
       self.radius = radius
       self.time = 0
       self.dead = False

    def is_interacting(self, obj):
        return self.radius + obj.radius >= math.sqrt(math.pow(self.x - obj.x, 2) + math.pow(self.y - obj.y, 2))

    def brownian_move(self):
        angle = random.random() * 2 * math.pi
        delta = 0.2
        self.x += delta * math.cos(angle)
        self.y += delta * math.sin(angle)

    def turn(self):  # Take a single turn of small timestep
        self.brownian_move()


class Food(IntObject):
    def __init__(self, x, y):
        super().__init__(1, x, y, 1)

    def make_protein(self):
        return Protein((random.random() * 10 - 5) + self.x, (random.random() * 10 - 5) + self.y, self, 0.1)  # 0.1 is a special type


class Protein(IntObject):
    def __init__(self, x, y, creator_obj, p_type):
        super().__init__(0, x, y, 2)
        self.p_type = p_type  # float that describes what type of protien it is
        self.creator_obj = creator_obj  # We don't want this protein to interact with the creator


class Roundling(IntObject):
    def __init__(self, energy, x, y, radius, model):
        super().__init__(energy, x, y, radius)
        self.model = model
        self.proteins = []

    def decide_move(self):  # 2 layers TODO figure out how tf works
        input_data = [self.x, self.y]
        p_frequency = []
        for p in self.proteins:
            found = False
            for f in p_frequency:
                if f[0] == p.p_type:
                    f[1] += 1
                    found = True
                    break
            if not found:
                p_frequency.append([p.p_type, 1])
        p_frequency.sort(key=lambda x: x[1], reverse=True)
        if len(p_frequency) < 5:
            iterv = len(p_frequency)
        else:
            iterv = 5

        for i in range(iterv):
            input_data.append(p_frequency[i][0])
            input_data.append(p_frequency[i][1])
        padding = 5 - iterv
        for i in range(padding):
            input_data.append(0)
            input_data.append(0)
        input_data.append(self.energy)
        return self.model.predict(tf.reshape(tf.convert_to_tensor(input_data), shape=(1, 13)))

    def move(self, output_vector):  # TODO remember to subtract energy according to an energy loss function per vector magnitude
        c = 0.3
        passive_energy = 0.01
        self.brownian_move()
        out = output_vector[0]
        self.x += out[0]
        self.y += out[1]
        dist_delta = math.sqrt(math.pow(out[0], 2) + math.pow(out[1], 2))
        if out[2] > 0.5: # create protein with x and y
            if out[3] > 5:
                protein_x = 5
            else:
                protein_x = out[3]
            if out[4] > 5:
                protein_y = 5
            else:
                protein_y = out[4]

            protein_cost = 0.05
            self.energy -= protein_cost
        self.energy -= dist_delta * c * self.radius + passive_energy # currently linear; we should experiment with exponential c
        if out[2] > 0.5:
            return Protein(protein_x + self.x, protein_y + self.y, self, out[2])
        else:
            return

    def turn(self):  # Takes a turn with timespan of single timestep; remember to subtract energy proportional to move, size, protein
        return self.move(self.decide_move())

    def reproduce(self):  # add random vector to weights and biases
        self.energy = self.energy / 2
        child_energy = self.energy
        child_x = (random.random() * 2 - 1) + self.x
        child_y = (random.random() * 2 - 1) + self.y
        child_radius_delta = (random.random() * 2 - 1)
        while child_radius_delta + self.radius <= 0:
            child_radius_delta = (random.random() * 2 - 1)

        child_radius = (random.random() * 2 - 1) + self.radius

        child_model = Sequential()
        child_model.add(Dense(16, activation='linear', input_shape=[13]))
        child_model.add(Dense(16, activation='linear'))
        child_model.add(Dense(5, activation='linear'))
        copy_and_randomize_weights(self.model, child_model)

        return Roundling(child_energy, child_x, child_y, child_radius, child_model)


if __name__ == "__main__":
    random.seed(2031239)
    ROUNDLING_COUNT = 70
    FOOD_COUNT = 40

    # initialize roundlings
    board = Board(100, 100, FOOD_COUNT)
    board.populate(ROUNDLING_COUNT)

    while True:
        board.step_time()
        print(f"Time elapsed: {board.time}")
        print(f"Roundling Count: {len(board.roundlings)}")
        print(f"Protein Count: {len(board.proteins)}")
        time.sleep(0.5)
