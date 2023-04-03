# context_encoders 图像修复的通用模型讲解 tf+keras
# 中航恒拓
# by Plusleft
# 2021.05.17
import glob
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import Activation, Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
# 分配GPU资源 我这里是2080的卡，分配了百分之三十显存 如果没有显卡可能会慢一些 但不影响程序运行
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

class ContextEncoder():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256

        self.mask_height = 64
        self.mask_width = 64

        self.channels = 3

        self.sum_classes = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)

        optimizer = Adam(0.0001, 0.5)

        # 生成器判别器
        self.generator = self.build_generator()

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # 准备联合训练
        self.discriminator.trainable = False
        masked_img = Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)
        valid = self.discriminator(gen_missing)
        self.combined = Model(masked_img, [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[0.999, 0.001],
                              optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        # 先定义编码器

        # 输入256*256*3的遮挡图
        model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # ——>128*12*64 特征图
        model.add(Conv2D(64, kernel_size=4, strides=2,  padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, kernel_size=4, strides=2,  padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 8*8*512

        # 进入中间层

        # Decoder ——>上采样＋卷积
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=4, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        # 第一层upconv完成 ——> 16*16*256

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        # 32*32*128

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        # 64*64*64

        model.add(Conv2D(self.channels, kernel_size=2, padding='same'))
        model.add(Activation('tanh'))

        # 64*64*3 输出完成

        model.summary()

        masked_img = Input(shape=self.img_shape)
        gen_missing = model(masked_img)

        return Model(masked_img, gen_missing)

    def build_discriminator(self):

        model = Sequential()

        # 64*64*3
        model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=self.missing_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 32*32*64

        model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 16*16*128

        model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 8*8*256

        model.add(Conv2D(512, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 4*4*512

        model.add(Flatten())
        # 16384

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(self.missing_shape)
        validity = model(img)
        # 完成了输入64*64*3 图片 输出真是概率

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        if os.path.exists('saved_model/discriminator_weights.hdf5') and os.path.exists(
                'saved_model/generator_weights.hdf5'):
            self.discriminator.load_weights('saved_model/discriminator_weights.hdf5')
            self.generator.load_weights('saved_model/generator_weights.hdf5')
            print('-------------load the model-----------------')

        X_train = []

        list = glob.glob(r'train_images/face2/*.jpg')
        for l in list:
            im = cv2.imread(l)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            X_train.append(im)
        X_train = np.array(X_train)

        print('X_train.shape', X_train.shape, "———————————————————数据集加载完成——————————")

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # 训练判别器
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            imgs = imgs / 175.5 - 1.  # -1 - 1
            # 随机抽取batchsize个真实图像

            masked_imgs, missing_parts, _ = self.mask_randomly(imgs)
            # masked_imgs就代表了遮挡的batch个图像
            # missing_parts就代表了丢失的batch个图像块

            gen_missing = self.generator.predict(masked_imgs)
            # 通过真假两个图训练判别器
            d_loss_real = self.discriminator.train_on_batch(missing_parts, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_missing, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # 返回损失值和准确率

            # 训练生成器
            g_loss = self.combined.train_on_batch(masked_imgs, [missing_parts, valid])
            # 在这里 使用的是 MSE ADloss

            # 打印损失值以及准确率
            print("%d [D loss: %f, acc: %.2f%%] [G loss mse: %f, ad loss: %f]" % (
                epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))
            # d_loss[0]判别器损失, d_loss[1]准确率, g_loss[0]联合模型的重建损失, g_loss[1]联合模型的对抗损失
            if epoch % sample_interval == 0:
                # 随机生成5个整数
                idx = np.random.randint(0, X_train.shape[0], 5)
                imgs = X_train[idx]
                imgs = imgs / 127.5 - 1.
                self.sample_images(epoch, imgs)
            if epoch % 1000 == 0:
                self.save_model()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            print(json_string)
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        # save(self.generator, "generator")
        # save(self.discriminator, "discriminator")

    def sample_images(self, epoch, imgs):
        r, c = 3, 5
        masked_imgs, missing_parts, (y1, y2, x1, x2) = self.mask_randomly(imgs)
        gen_missing = self.generator.predict(masked_imgs)

        imgs = 0.5 * imgs + 0.5
        masked_imgs = 0.5 * masked_imgs + 0.5
        gen_missing = 0.5 * gen_missing + 0.5

        fig, axs = plt.subplots(r, c)
        # imshow 绘制原图 遮挡图 和修复图
        for i in range(c):
            axs[0, i].imshow(imgs[i, :, :])
            axs[0, i].axis('off')
            axs[1, i].imshow(masked_imgs[i, :, :])
            axs[1, i].axis('off')
            filled_in = imgs[i].copy()
            filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
            axs[2, i].imshow(filled_in)
            axs[2, i].axis('off')
        fig.savefig("images/%d.png" % epoch, dpi=256)
        plt.close()

    def mask_randomly(self, imgs):
        y1 = np.random.randint(0, self.img_rows-self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width

        # 复制原图 待遮挡（这里只需要完成像素置0即可完成遮挡）
        masked_imgs = np.empty_like(imgs)

        # 丢失区域内容大小尺寸定义完毕 （这里只需要将丢失的像素点复制进来）
        missing_parts = np.empty((imgs.shape[0], self.mask_height, self.mask_width, self.channels))

        for i, img in enumerate(imgs):
            masked_img = img.copy()  # 首先复制原图 也就是准备完成遮挡单个图
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]  # 随机生成的每个遮挡坐标
            missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
            masked_img[_y1:_y2, _x1:_x2, :] = 0  # 置0操作 完成遮挡
            masked_imgs[i] = masked_img  # 存入 masked_imgs
        return masked_imgs, missing_parts, (y1, y2, x1, x2)


if __name__ == '__main__':
    context_encoder = ContextEncoder()
    context_encoder.train(epochs=100, batch_size=16, sample_interval=50)

