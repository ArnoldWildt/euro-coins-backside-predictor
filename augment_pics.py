import Augmentor

p = Augmentor.Pipeline("./new_pics/")

p.greyscale(probability=1)
p.rotate(probability=0.33, max_left_rotation=10, max_right_rotation=10)
p.rotate_random_90(probability=0.33)
p.random_distortion(probability=0.33, grid_height=10,grid_width=10, magnitude=10)
p.resize(probability=1, width=200, height=200)

p.sample(10000)
