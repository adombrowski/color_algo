from PIL import Image
import requests
from io import BytesIO
import numpy as np
from functools import reduce
from operator import mul
from sklearn.cluster import KMeans


"""
image_utils.py holds utility functions related to image
manipulation.
"""

class Pic:
	def __init__(self, img_path):
		def scrapeImg(url):
			response = requests.get(url)
			return BytesIO(response.content)
		def loadImg(path):
			return Image.open(path)
		def genRGB(pix, dim):
			return np.array([pix[l,w] for l in range(0,dim[0]) for w in range(0,dim[1])])

		##load img
		if img_path.startswith("http"):
			##replace https with http
			url = img_path.replace("https", "http")
			##request image
			self._img = loadImg(scrapeImg(url))
		else:
			self._img = loadImg(img_path)

		##store img attributes
		self.size = self._img.size

		##generate rgb mattrix
		self._rgb = genRGB(self._img.load(), self.size)

	def genPalette(self, count=8):
		"""
		genPalette() runs kmeans clustering using sklearn
		library to generate centroid values. Outputs 
		cluster centroid values as np.array

		:param count: number of palette elements, user defined
		"""

		return KMeans(n_clusters=count, random_state = 0).fit(self._rgb).cluster_centers_
