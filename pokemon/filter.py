from PIL import Image
import os

delete_list = []

def pic_filter(path):
# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    number_list = os.listdir(path)
    for number in number_list:
        number_path = os.path.join(path, number)
        pic_list = os.listdir(number_path)
        for pic in pic_list:
            pic_path = os.path.join(number_path, pic)
            try:
                tmp = Image.open(pic_path)
                tmp.close()
            except Exception as err:
                delete_list.append(pic_path)

    # with open(path, 'rb') as f:
    #     print(f)
        # with Image.open(f) as img:
        #     return img.convert('RGB')
path = "/home/qh/git/pytorch_test/pokemon/data/train/"
pic_filter(path)
for d_file in delete_list:
    os.remove(d_file)
# print(delete_list)