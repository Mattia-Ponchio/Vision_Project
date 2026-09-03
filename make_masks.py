import argparse
import numpy as np
import numpy.random as npr
import random
from PIL import Image

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]] # this is the list of possible actions for the random walk: right, left, down, up

"""
Common inputs to the functions:
- canvas: a 2D numpy array representing the image mask, initialized with ones (1s) indicating untouched areas
- ini_x, ini_y: the initial x and y coordinates on the canvas where the mask generation starts
- borders: the number of pixels from the edges that should remain untouched (1s)
- looping: a boolean flag indicating whether the mask generation can go beyond the borders of the canvas
"""

# This function performs a random walk on the canvas starting from (ini_x, ini_y) for a given length. 
# The walk can either loop around the borders or be constrained within them based on the looping parameter.

# version of rw where length is the number of steps to take in the random walk
def random_walk(canvas, ini_x, ini_y, length, borders, looping=False):
	img_size = canvas.shape[-1]
	r = npr.randint(low = 0, high = len(action_list), size = length)
	steps = np.array([action_list[i] for i in r])
	if looping:
		valid_range = img_size - 2 * borders
		# If looping then there are no "walls", so every position can be computed immediately with the cumulative sum
		# without waiting to generate the previous step
		cum_steps_x = np.cumsum(steps[:, 0])
		cum_steps_y = np.cumsum(steps[:, 1])
		# Recentering without borders
		raw_x = (ini_x - borders) + cum_steps_x
		raw_y = (ini_y - borders) + cum_steps_y
		
		# Looping + borders
		x_list = (raw_x % valid_range) + borders
		y_list = (raw_y % valid_range) + borders
		
		canvas[x_list.astype(int), y_list.astype(int)] = 0
		
	else:
		x_list = np.zeros(length, dtype=int)
		y_list = np.zeros(length, dtype=int)
		x = ini_x
		y = ini_y
		for i in range(length):
			direction = steps[i]
			x = np.clip(x + direction[0], a_min=borders, a_max=img_size - 1 - borders)
			y = np.clip(y + direction[1], a_min=borders, a_max=img_size - 1 - borders)
			x_list[i] = x
			y_list[i] = y
		canvas[x_list, y_list] = 0
		
	return canvas # it outputs the mask where 0s represent the path of the walk and 1s represent untouched areas.


# version of rw where target_ratio is exaclty the ratio of pixels to be corrupted
def random_walk_until_ratio(canvas, ini_x, ini_y, target_ratio, borders):
    img_size = canvas.shape[-1]
    target = int(img_size * img_size * target_ratio)

    visited = set()

    x = int(ini_x)
    y = int(ini_y)

    # enforce borders on start
    x = np.clip(x, borders, img_size - 1 - borders)
    y = np.clip(y, borders, img_size - 1 - borders)

    visited.add((x, y))
    canvas[x, y] = 0

    while len(visited) < target:
        dx, dy = action_list[npr.randint(0, 4)]

        x_new = x + dx
        y_new = y + dy

        # enforce borders
        x = np.clip(x_new, borders, img_size - 1 - borders)
        y = np.clip(y_new, borders, img_size - 1 - borders)

        visited.add((x, y))
        canvas[x, y] = 0

    return canvas


# This function adds salt and pepper noise to the canvas. 
# It randomly selects a number of pixels based on the specified ratio and sets them to 0 (black), simulating noise in the image.
# The 'ratio' input is the fraction of the total number of pixels in the image to corrupt
def salt_and_pepper(canvas, borders, ratio):
	img_size = canvas.shape[-1]
	noise_pixels = int(img_size * img_size * ratio)
	x = npr.randint(low = borders, high = img_size - 1 - borders, size = noise_pixels)
	y = npr.randint(low = borders, high = img_size - 1 - borders, size = noise_pixels)
	canvas[x, y] = 0
	return canvas # it outputs the mask where 0s represent the noisy pixels and 1s represent untouched areas.


# This function creates a box-shaped region on the canvas.
# The 'area' input specifies the total number of pixels to be set to 0 (black) in the box, and the function calculates the dimensions of the box accordingly.
def box(canvas, ini_x, ini_y, borders, area, looping=False):
	img_size = canvas.shape[-1]
	base = npr.randint(low=1, high=img_size)
	height = np.clip(area//base, 1, img_size)
    
    # Starting point is the vertex in the corner of the box
    # 0 = top left; 1 = top right; 2 = bot left; 3 = bot right
	orientation = npr.randint(0, 4)

	# If orientation in {0,2}: left side so move towards the right on x, otherwise to the left
	# If orientation in {2,3}: bottom side so move upwards on y, otherwise downwards
	step_x = 1 if orientation in [0, 2] else -1
	step_y = 1 if orientation in [2, 3] else -1

	# Indices
	x_indices = ini_x + step_x * np.arange(base)
	y_indices = ini_y + step_y * np.arange(height)

	if looping:
		# Window size for valid range
		valid_range = img_size - 2 * borders
		
		# Looping by starting from the border on the opposite side when crossing the border
		x_indices = ((x_indices - borders) % valid_range) + borders
		y_indices = ((y_indices - borders) % valid_range) + borders
	else:
		# No looping: no border crossing allowed
		x_indices = np.clip(x_indices, borders, img_size - 1 - borders)
		y_indices = np.clip(y_indices, borders, img_size - 1 - borders)

	canvas[np.ix_(x_indices, y_indices)] = 0

	return canvas # it outputs the mask where 0s represent the box region and 1s represent untouched areas.


# this function creates a circular region on the canvas.
# The 'radius' input specifies the radius of the circle
def circles(canvas, ini_x, ini_y, radius, borders, looping=False):
    img_size = canvas.shape[-1]
    
    # Indices
    y_indices, x_indices = np.ogrid[:img_size, :img_size]
    
    if looping:
        # Window size for valid range
        valid_range = img_size - 2 * borders
        
        # Pixel-center distance
        dx = x_indices - ini_x
        dy = y_indices - ini_y
        
        # Distances taking into account border + loop
        dx = (dx + valid_range / 2) % valid_range - valid_range / 2
        dy = (dy + valid_range / 2) % valid_range - valid_range / 2
        
        # R^2 mask
        distance_squared = dx**2 + dy**2
        circular_mask = distance_squared <= radius**2
        
		# Border mask
        valid_zone = (x_indices >= borders) & (x_indices < img_size - borders) & \
                     (y_indices >= borders) & (y_indices < img_size - borders)
        
        # Border + loop mask
        canvas[circular_mask & valid_zone] = 0
        
    else:
        distance_squared = (x_indices - ini_x)**2 + (y_indices - ini_y)**2
        circular_mask = distance_squared <= radius**2
        
        # Clipping at the borders
        valid_zone = (x_indices >= borders) & (x_indices < img_size - borders) & \
                     (y_indices >= borders) & (y_indices < img_size - borders)
        
        canvas[circular_mask & valid_zone] = 0
        
    return canvas # it outputs the mask where 0s represent the circular region and 1s represent untouched areas.

	

if __name__ == '__main__':
	import os

	parser = argparse.ArgumentParser()
	parser.add_argument('--image_size', type=int, default=256, help = "Dimension for squared image, default 256")
	parser.add_argument('--N', type=int, default=1000, help = "Number of masks to create, default 1000")
	parser.add_argument('--borders', type=int, default=0, help = "Number of pixels from the borders that can be corrupted, default 0")
	parser.add_argument('--save_dir', type=str, default='masks', help = "Saving directory path, default ./masks")
	parser.add_argument('--mode', type=str, default="all", 
			help = "Mask type:\n box - for rectangular boxes || sap - for salt and pepper noise || circle - for circular masks || all - for random walk masks with the default ratios of noise || rw - for random walk masks with user defined ratio of noise ")
	parser.add_argument('--ratio', type=float, default=0.2, help = "Fraction of the total number of pixels in the image to corrupt, default 0.2")
	parser.add_argument('--loop', action='store_true', help="Whether looping is allowed (include flag to enable)")
	args = parser.parse_args()
	# example: python make_masks.py --N 10 --mode circle --ratio 0.2 --loop
	# python make_masks.py --N 40 --save_dir masks/new_masks --mode sap --ratio 0.1 

	default_ratios = [ [0.01, 0.10], [0.10, 0.20], [0.20, 0.30], [0.30, 0.40], [0.40, 0.50], [0.50, 0.60] ]

	npr.seed(seed=2026)

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
        
	match args.mode:		
		case "sap":
			for i in range(args.N):
				canvas = np.ones((args.image_size, args.image_size)).astype("i")
				mask = salt_and_pepper(canvas, args.borders, args.ratio)
				# print("save:", i, np.sum(mask))

				img = Image.fromarray(mask * 255).convert('1')
				img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))
		
		case "box":
			ini_x_list = npr.randint(0+args.borders, args.image_size - 1 - args.borders, args.N)
			ini_y_list = npr.randint(0+args.borders, args.image_size - 1 - args.borders, args.N)
			area = int( args.image_size**2 * args.ratio )
			for i in range(args.N):
				canvas = np.ones((args.image_size, args.image_size)).astype("i")
				ini_x = ini_x_list[i]
				ini_y = ini_y_list[i]
				mask = box(canvas, ini_x, ini_y, args.borders, area, args.loop)
				#print("save:", i, np.sum(mask))

				img = Image.fromarray(mask * 255).convert('1')
				img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))
		
		case "circle":
			ini_x_list = npr.randint(0+args.borders, args.image_size - 1 - args.borders, args.N)
			ini_y_list = npr.randint(0+args.borders, args.image_size - 1 - args.borders, args.N)
			radius = int( np.sqrt ( args.image_size * args.image_size * args.ratio / np.pi ) )
			for i in range(args.N):
				canvas = np.ones((args.image_size, args.image_size)).astype("i")
				ini_x = ini_x_list[i]
				ini_y = ini_y_list[i]

				mask = circles(canvas, ini_x, ini_y, radius, args.borders, args.loop)
				#print("save:", i, np.sum(mask))

				img = Image.fromarray(mask * 255).convert('1')
				img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))
				
		case "all":
			num_elem_per_ratio = args.N // len(default_ratios)
			Ns = [ num_elem_per_ratio for i in range(len(default_ratios) - 1) ]
			Ns.append( args.N - sum(Ns) )
			for j, num_elem in enumerate(Ns):
				length = (args.image_size**2 * npr.uniform(low = default_ratios[j][0], high = default_ratios[j][1], size = num_elem) ).astype("i")
				ini_x_list = npr.randint(0+args.borders, args.image_size - 1 - args.borders, num_elem)
				ini_y_list = npr.randint(0+args.borders, args.image_size - 1 - args.borders, num_elem)
				for i in range(num_elem):
					canvas = np.ones((args.image_size, args.image_size)).astype("i")
					ini_x = ini_x_list[i]
					ini_y = ini_y_list[i]
					mask = random_walk(canvas, ini_x, ini_y, length[i], args.borders, args.loop)
					#print("save:", i, np.sum(mask))

					img = Image.fromarray(mask * 255).convert('1')
					img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i + j*Ns[0]))
		
		case "rw":
			length = int ( args.image_size * args.image_size * args.ratio)
			ini_x_list = npr.randint(0+args.borders, args.image_size - 1 - args.borders, args.N)
			ini_y_list = npr.randint(0+args.borders, args.image_size - 1 - args.borders, args.N)
			
			for i in range(args.N):
				canvas = np.ones((args.image_size, args.image_size)).astype("i")
				ini_x = ini_x_list[i]
				ini_y = ini_y_list[i]
				mask = random_walk(canvas, ini_x, ini_y, length, args.borders, args.loop)
				#print("save:", i, np.sum(mask))

				img = Image.fromarray(mask * 255).convert('1')
				img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))

		# added case for random walk with ratio as the number of steps
		case "rw_ratio":
			ini_x_list = npr.randint(0 + args.borders,	args.image_size - 1 - args.borders,	args.N)
			ini_y_list = npr.randint(0 + args.borders,	args.image_size - 1 - args.borders,	args.N)

			for i in range(args.N):
				canvas = np.ones((args.image_size, args.image_size)).astype("i")

				ini_x = ini_x_list[i]
				ini_y = ini_y_list[i]

				mask = random_walk_until_ratio(
					canvas,
					ini_x,
					ini_y,
					args.ratio,
					args.borders
				)

				img = Image.fromarray(mask * 255).convert('1')
				img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))
