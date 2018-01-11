from PIL import Image
import os
import warnings

warnings.filterwarnings("ignore")

delete_list = []

def picPath_filter(path):
    """
    找出不能用Image.open打开的错误文件，并删除
    """
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

path = "/home/qh/data/train"
picPath_filter(path)
count = 0
print(delete_list)
for d_file in delete_list:
    count = count + 1
    os.remove(d_file)
print("delete count: " + str(count))



# def pic_transfer(path):
# """
#     图片格式统一转换
# """
# from PIL import Image
# im = Image.open(r"C:\jk.png")
# bg = Image.new("RGB", im.size, (255,255,255))
# bg.paste(im,im)
# bg.save(r"C:\jk2.jpg")

#     with open(path, 'rb') as f:
#         with Image.open(f) as img:
#             try:
#                 return img.convert('RGB')
#             except:
#                 return img.convert('RGBA')
# import warnings
# # 忽略警告
# # warnings.filterwarnings("ignore")

# # 捕捉警告
# warnings.filterwarnings('error')

# def picFile_filter(filepath):
#     """
#     找出不能用Image.open打开的错误文件，并删除
#     """
# # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     try:
#         img = Image.open(filepath)
#         img.convert('RGB')
#         img.close()
#     except UserWarning as e:
#         delete_list.append(filepath)


# picFile_filter("/home/qh/test/data/train/100/63.jpg")
# picFile_filter("/home/qh/test/data/train/100/64.png")
# print(delete_list)

# def picPath_filter(path):
#     """
#     找出不能用Image.open打开的错误文件，并删除
#     """
# # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     number_list = os.listdir(path)
#     for number in number_list:
#         number_path = os.path.join(path, number)
#         pic_list = os.listdir(number_path)
#         for pic in pic_list:
#             pic_path = os.path.join(number_path, pic)
#             try:
#                 img = Image.open(pic_path)
#                 # img.convert('RGB')
#                 img.close()
#             except UserWarning as e:
#                 delete_list.append(pic_path)
# path = "/home/qh/data/train"
# picPath_filter(path)
# count = 0
# print("delete count: " + str(count))
# print(delete_list)

# for d_file in delete_list:
#     count = count + 1
#     os.remove(d_file)
