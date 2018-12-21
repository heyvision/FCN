import numpy as np
import PIL.Image as Image
img = Image.open('./logs/pred_6.png')
src = Image.open('./logs/inp_6.png')
mat = np.array(img)
mat_src = np.array(src)
for i in range(mat.shape[0]):
	for j in range(mat.shape[1]):
		if mat[i,j] != 0:
			mat_src[i,j]=200
src = Image.fromarray(np.uint8(mat_src))
src.show()