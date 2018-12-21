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
    usage: python ./json2png.py json_file
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