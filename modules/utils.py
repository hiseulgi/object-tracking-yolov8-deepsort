import random


def random_pastel_color():
    red = random.randint(128, 255)
    green = random.randint(128, 255)
    blue = random.randint(128, 255)
    return (blue, green, red)
