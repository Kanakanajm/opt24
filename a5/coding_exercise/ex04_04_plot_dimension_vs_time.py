import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import subprocess

mpl.use('TkAgg') # probably not everyone requires this, feel free to remove it


img_size_min = 12
img_size_max = 93

img_sizes_to_check = range(img_size_min, img_size_max, 10)

time_values = []
time_values_1 = []
for i in img_sizes_to_check:

	command_to_exec = ['python', 'ex04_03_image_deblurring.py',
					'--img_size=' + str(i),
         			'--run_exp_option=' + str(1)]

	print("Command executing is " + " ".join(command_to_exec))

    # Execute the command in command_to_exec onto the terminal 
    # from python itself (Hint: Check subprocess package details.)

	subprocess.run(command_to_exec)

    
	try:
		# load files for Newton method and image size i (check NewtonMethod.py)
		a = np.loadtxt('results/newton_time_'+ str(i) + '.txt')
		time_taken = a
	except:
		time_taken = 20

	time_values += [time_taken]

	try:
		# load files for Gradient Descent method and image size i  (check GradientDescent.py)
		a = a = np.loadtxt('results/cg_time_'+ str(i) + '.txt')
		time_taken = a
	except:
		time_taken = 20  # maximum time

	time_values_1 += [time_taken]

max_line = np.ones_like(img_sizes_to_check)*20
fig = plt.figure()
plt.plot(img_sizes_to_check, time_values, color='blue',
         label='TIME TAKEN - Newton (1 iteration)')
plt.plot(img_sizes_to_check, time_values_1, color='green',
         label='TIME TAKEN - GD (100 iterations)')
plt.plot(img_sizes_to_check, max_line, color='red', label='MAX ALLOWED TIME')
plt.title('Algorithms Behavior with respect to Dimension')
plt.legend()
plt.ylabel('Time taken')
plt.xlabel('Image dimension')
plt.show()
