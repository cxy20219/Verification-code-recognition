import random
import time
from captcha.image import ImageCaptcha
captcha_array = list("1234567890qwertyuiopasdfghjklzxcvbnm")
captcha_size = 4
if __name__ == "__main__":
    for i in range(1000):
        image=ImageCaptcha()

        # 随机抽4个元素
        image_text="".join(random.sample(captcha_array,captcha_size))

        # 加时间戳防止重复
        # image_path="datasets/train/{}_{}.png".format(image_text,int(time.time()))
        image_path="datasets/test/{}_{}.png".format(image_text,int(time.time()))

        image.write(image_text,image_path)

        print("正在生成第{}张".format(i+1))