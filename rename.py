# import glob
# from PIL import Image
#
# # 获取指定目录下的所有图片  D:\PyCharm Community Edition 2021.1.1\worksapce\context_encoder\images
# print(glob.glob(r"../context_encoder/images/*.jpg"))
#
# # 获取上级目录的所有.py文件
# im=Image.open(r'../context_encoder/images/000001.jpg')
# im.thumbnail(256,256)
#
# print(im.format,im.size,im.mode)



import os
import glob
from PIL import Image

def resize_pic(inputDir, outDir):
    # 提取指定目录下的所有图片的全路径名
    resourcesDirs = glob.glob(inputDir)
    for filePath in resourcesDirs:
        print(filePath)
        img = Image.open(filePath)
        # 重置图像分辨率
        # new_img=im.resize((width, height),Image.BILINEAR)
        print(img.format, img.size, img.mode)
        # os.path.basename 从文件的全路径名中获取文件名
        fileName = os.path.basename(filePath)
        fileName=fileName[:-4]
        print(fileName)
        # os.path.join用来拼接字符串
        out=os.path.join(outDir, fileName)
        img.save(out)
    print('Done!')

if __name__ == '__main__':
    inputDir="D:/PyCharm Community Edition 2021.1.1/worksapce/Region-wise-Inpainting-master/dataset/celeba-64/*.jpg"
    outDir="../context_encoder/outDir/"
    resize_pic(inputDir, outDir)