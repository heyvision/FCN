# 在tensorflow上使用FCN训练自己的数据
 
参考文档：https://blog.csdn.net/m0_37407756/article/details/83379026　tensorflow实现FCN完成训练自己标注的数据

博客：https://blog.csdn.net/qq_42143583/article/details/85164522

## 准备数据
 1. 在github上下载fcn的tensorflow版本[实现](https://github.com/shekkizh/FCN.tensorflow)
 2. 下载[labelme](https://github.com/wkentaro/labelme)对自己的图片进行标注，把生成的json文件放入你的json_file中
 3. 使用json2png.py批量将json文件转化为可以进行训练的png格式图片，此时生成的文件夹下每个json文件夹对应img.png,lable.png,label_viz.png三张图片。img.png作为输入，label.png作为标注图像（显示为全黑）。程序中将生成的png图片转化成8位的图片存储，此时label.png中像素实际以0,1,2...来分割图像，可以将灰度值放大来进行验证。
```python
import argparse
import json
import os
import os.path as osp
import warnings
 
import numpy as np
import PIL.Image
 
from labelme import utils
 
def main():
    '''
    usage: python json2png.py json_file
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()
 
    json_file = args.json_file
 
    list = os.listdir(json_file)
    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])
        if os.path.isfile(path):
            data = json.load(open(path))
            img = utils.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
 
            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            lbl_viz = utils.draw_label(lbl, img, captions)
            out_dir = osp.basename(list[i]).replace('.', '_')
            # out_dir = osp.join(osp.dirname(list[i]), out_dir)
            out_dir = osp.join('./png', out_dir)
            if not osp.exists(out_dir):
                os.mkdir(out_dir)
            
            PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
            lbl = PIL.Image.fromarray(np.uint8(lbl))
            lbl.save(osp.join(out_dir, 'label.png'))
            # PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
            print('Saved to: %s' % out_dir)
 
if __name__ == '__main__':
    main()
```
 4. 将生成的img和label按原数据集[http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)中MIT_SceneParsing/ADEChallengeData2016文件夹中类似存储。下面是批量存入annotations/training和images/training中的代码，validation中的图片可以从training中剪切，也可以修改下面的代码。
```python
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
```
## 训练
 1. 将FCN.py中NUM_OF_CLASSESS改为自己训练数据的类别数，注意加上背景。vgg-19预训练模型在程序运行中会进行下载，也可以在训练前下载好[http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)放在Model_zoo文件夹中。将data_dir改为自己的数据集所在文件，训练时mode为train。
 2. read_MITSceneParsingData.py中将SceneParsing_folder令为自己的文件夹`SceneParsing_folder = 'myData'`，而不是下载。
3. 默认batch_size大小是2，迭代次数为10000次
4. 损失函数可视化：
 ```bash
 tensorboard --logdir ./logs/train
 ```

## 测试
在FCN.py中将mode改为visualize，将网络生成的预测图像中灰度值不为0的点，可以在原图上对应位置将其灰度值修改为某固定值如200，就完成了可视化。FCN.py中修改如下。
```python
elif FLAGS.mode == "visualize":
	#num: the number of images to be tested which can be a single batch_size or all validation set
    valid_images, valid_annotations, num = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
    pred = sess.run(pred_annotation, feed_dict={image: valid_images, keep_probability: 1.0})
    pred = np.squeeze(pred, axis=3)
	for itr in range(num):
	    src_img = valid_images[itr].astype(np.uint8)
	    pred_img = pred[itr].astype(np.uint8)
	    #save images to ./logs/test_visualize
	    utils.save_image(src_img, FLAGS.logs_dir + 'test_visualize/', name="inp_" + str(itr))
	    utils.save_image(pred_img, FLAGS.logs_dir + 'test_visualize/', name="pred_" + str(itr))
	    for i in range(pred_img.shape[0]):
	        for j in range(pred_img.shape[1]):
	            if pred_img[i,j] != 0:
	            	#if your source images are RGB format, you need to change three channels
	                src_img[i,j]=200
	    utils.save_image(src_img, FLAGS.logs_dir + 'test_visualize/', name="visual_" + str(itr))
	    print("Saved image: %d" % itr)
```
