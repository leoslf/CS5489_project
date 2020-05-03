from PIL import Image
import os, sys
import glob
from multiprocessing import Pool

target_shape = (299, 299)

def preprocess(filename):
    if os.path.isfile(filename):
        im = Image.open(filename)
        basename = os.path.basename(filename)
        f, e = os.path.splitext(basename)
        imResize = im.resize(target_shape, Image.ANTIALIAS)
        imResize.save("preprocessed/%s.png" % f) # , 'PNG', quality=90)

if __name__ == "__main__":
    with Pool() as pool:
        result = pool.map_async(preprocess, glob.glob("images/*.jpg"))
        print (result.get())


