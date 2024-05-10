#!/usr/bin/env python3
import enum
import os
import time
from typing import List

import numpy as np
try:
    from sense_hat import SenseHat
except ModuleNotFoundError:
    print('unable to import sense_hat. Trying to import sense_emu.')
    from sense_emu import SenseHat
import cv2


class Mode(enum.Enum):
    Image = 0
    BrickGame = 1
    Temperature = 2


class GameMode(enum.Enum):
    InGame = 0
    Won = 1
    Lost = 2


class Main:
    def __init__(self):
        self.sense = SenseHat()
        self.images = load_scaled_images((8, 8))
        self.current_image_index = 0
        self.game_state = GameState()
        self.mode = Mode.Image

    def run(self):
        while True:
            for event in self.sense.stick.get_events():
                self.handle_event(event)
            self.tick()
            self.render()
            time.sleep(0.05)

    def render(self):
        if self.mode == Mode.Image:
            image = self.images[self.current_image_index]
            modified_image = change_hsv(image, saturation=self._get_saturation(), hue=self._get_hue())
            self._show_image(modified_image)
        elif self.mode == Mode.BrickGame:
            if self.game_state.game_mode == GameMode.InGame:
                image = self.game_state.get_pixels()
                self._show_image(image)
            elif self.game_state.game_mode == GameMode.Lost:
                self.sense.show_message('Verloren :(')
                time.sleep(0.5)
                self.mode = Mode.Temperature
            elif self.game_state.game_mode == GameMode.Won:
                self.sense.show_message('Gewonnen :)')
                time.sleep(0.5)
                self.mode = Mode.Temperature
        elif self.mode == Mode.Temperature:
            temp = max(int(round(self.sense.get_temperature())), 0)
            temp_str = str(temp)
            pixels = np.zeros((8, 8, 3))
            digit1 = digit_to_np(int(temp_str[0]))
            pixels[1:6, :3, :] = digit1
            if temp >= 10:
                digit2 = digit_to_np(int(temp_str[1]))
                pixels[1:6, 4:7, :] = digit2
            self._show_image(pixels.astype(np.uint8))

    def tick(self):
        if self.mode == Mode.BrickGame:
            self.game_state.tick()

    def _get_saturation(self):
        pitch = self.sense.get_accelerometer()['pitch']
        if pitch > 180:
            pitch = -(360 - pitch)
        saturation = np.clip((pitch + 45) / 60, 0, 2)
        return saturation

    def _get_hue(self):
        roll = self.sense.get_accelerometer()['roll']
        return roll / 360 * 255

    def _show_image(self, image):
        pixel_list = image.reshape((8 * 8, 3))
        self.sense.set_pixels(pixel_list)

    def handle_event(self, event):
        if self.mode == Mode.Image:
            if event.action == 'released' and event.direction == 'middle':
                self.mode = Mode.BrickGame
                self.game_state = GameState()
            elif event.action == 'released':
                if event.direction in ('left', 'down'):
                    self.current_image_index = (self.current_image_index - 1) % len(self.images)
                if event.direction in ('right', 'up'):
                    self.current_image_index = (self.current_image_index + 1) % len(self.images)
        elif self.mode == Mode.BrickGame:
            if event.action == 'released' and event.direction == 'middle':
                self.mode = Mode.Temperature
            if event.action in ('pressed', 'held') and event.direction in ('left', 'right'):
                self.game_state.handle_event(event.direction)
        elif self.mode == Mode.Temperature:
            if event.action == 'released' and event.direction == 'middle':
                self.mode = Mode.Image


class GameState:
    def __init__(self):
        self.player_position: int = 3
        self.ball_position: np.ndarray = np.array([3.5, 6.0])
        self.ball_speed = 0.1
        self.ball_direction: np.ndarray = np.array([0.027, -0.1])
        self.ball_direction *= (self.ball_speed / np.linalg.norm(self.ball_direction, ord=2))
        self.bricks = np.ones(8)
        self.game_mode = GameMode.InGame

    def tick(self):
        if self.game_mode == GameMode.InGame:
            self.ball_position += self.ball_direction

            # check brick collision
            if self.ball_position[1] <= 0.5:
                x_index_low = np.clip(int(self.ball_position[0]), 0, 7)
                if abs(self.ball_position[0] - x_index_low) < 0.7:
                    if self.bricks[x_index_low] == 1:
                        self.ball_direction[1] = abs(self.ball_direction[1])
                    self.bricks[x_index_low] = 0
                x_index_high = np.clip(int(self.ball_position[0])+1, 0, 7)
                if (self.ball_position[0] - x_index_high) < 0.7:
                    if self.bricks[x_index_high] == 1:
                        self.ball_direction[1] = abs(self.ball_direction[1])
                    self.bricks[x_index_high] = 0
                if np.sum(self.bricks) == 0:
                    self.game_mode = GameMode.Won
            if self.ball_position[1] <= -0.5:
                self.ball_direction[1] = abs(self.ball_direction[1])

            # check left right collision
            if self.ball_position[0] > 7.5:
                self.ball_direction[0] = -np.abs(self.ball_direction[0])
            if self.ball_position[0] <= -0.5:
                self.ball_direction[0] = np.abs(self.ball_direction[0])

            # check player collision
            if self.ball_position[1] >= 6:
                player_center = self.player_position + 0.5
                diff = self.ball_position[0] - player_center
                if abs(diff) < 1.2:
                    self.ball_direction[1] = -abs(self.ball_direction[1])
                    self.ball_direction[0] = diff * 0.1
                    self.ball_direction *= (self.ball_speed / np.linalg.norm(self.ball_direction, ord=2))

            # check lost
            if self.ball_position[1] >= 7.5:
                self.game_mode = GameMode.Lost

    def get_pixels(self):
        pixels = np.zeros((8, 8, 3))

        if self.game_mode == GameMode.InGame:
            # draw bricks
            brick_pixels = self.bricks.reshape((1, 8, 1)) * np.array([255, 255, 0]).reshape((1, 1, 3))
            pixels[0, :, :] = brick_pixels

            # draw ball
            ball_pos = np.clip(np.round(self.ball_position).astype(np.int32), 0, 7)
            pixels[ball_pos[1], ball_pos[0], :] = np.array([255, 0, 0])

            # draw player
            player_pos = np.clip(self.player_position, 0, 7)
            pixels[7, player_pos:player_pos+2, :] = np.array([0, 255, 0])

        return pixels.astype(np.uint8)

    def handle_event(self, direction):
        direction = -1 if direction == 'left' else 1
        self.player_position = np.clip(self.player_position + direction, 0, 6)


def load_scaled_images(target_size) -> List[np.ndarray]:
    images = []
    image_dir = 'images'
    for filename in os.listdir(image_dir):
        path = os.path.join(image_dir, filename)
        if path.endswith('.jpg') or path.endswith('.png'):
            image = cv2.imread(path)
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
    return images


def rotate_image(image, angle):
    """
    From https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def change_hsv(image, hue=0, saturation=1, value=1):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int32)
    if hue:
        hsv_image[..., 0] = hsv_image[..., 0] + hue
        hsv_image[hsv_image < 0] += 255
        hsv_image[hsv_image > 255] -= 255
    if saturation != 1:
        hsv_image[..., 1] = hsv_image[..., 1] * saturation
    if value != 1:
        hsv_image[..., 2] = hsv_image[..., 2] * value
    hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def digit_to_np(digit: int) -> np.ndarray:
    if digit == 0:
        arr = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])
    elif digit == 1:
        arr = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ])
    elif digit == 2:
        arr = np.array([
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
        ])
    elif digit == 3:
        arr = np.array([
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ])
    elif digit == 4:
        arr = np.array([
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ])
    elif digit == 5:
        arr = np.array([
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ])
    elif digit == 6:
        arr = np.array([
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])
    elif digit == 7:
        arr = np.array([
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ])
    elif digit == 8:
        arr = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])
    elif digit == 9:
        arr = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ])
    else:
        raise ValueError('Not a digit: {}'.format(str(digit)))
    return arr.reshape((*arr.shape, 1)) * np.array([255, 255, 255]).reshape(1, 1, 3)


if __name__ == '__main__':
    main = Main()
    main.run()

