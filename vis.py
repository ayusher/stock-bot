import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

model = tf.keras.models.load_model("models/"+sys.argv[1])
filter = model.layers[2].get_weights()[0]

fig, axes = plt.subplots(16, 8)

for a in range(128):
	ax = axes.flatten()[a]
	ax.imshow(np.reshape(filter, (128, 5, 3))[a])
	ax.set_xticks([])
	ax.set_yticks([])
plt.savefig("vis.png")
#print(np.reshape(filter, (128, 5, 3)).shape)