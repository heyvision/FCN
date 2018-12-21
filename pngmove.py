import os
import shutil
filedir = './png'
outdir_a = './FCN.tensorflow-master/Data_zoo/power/myData/annotations'
outdir_i = './FCN.tensorflow-master/Data_zoo/power/myData/images'
filelists = os.listdir(filedir)
filelists.sort()
for i,filename in enumerate(filelists):
	print(filename)
	filename = os.path.join(filedir, filename)

	shutil.move(os.path.join(filename,'label.png'),os.path.join(outdir_a,'training'))
	name1 = 'train-' + str(i+1).zfill(3) + '.png'
	os.rename(os.path.join(outdir_a,'training/label.png'),os.path.join(outdir_a,'training',name1))

	shutil.move(os.path.join(filename,'img.png'),os.path.join(outdir_i,'training'))
	name2 = 'train-' + str(i+1).zfill(3) + '.png'
	os.rename(os.path.join(outdir_i,'training/img.png'),os.path.join(outdir_i,'training',name2))