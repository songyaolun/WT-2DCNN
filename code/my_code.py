from tensorflow.keras.layers import Dense, Input, Conv1D, Conv2D, GlobalAveragePooling2D, MaxPool2D, AvgPool2D, \
    GlobalAveragePooling1D
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import sys
import os

import mlwt
import common
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


# deep 1DCNN
def SimpleCNN():
    wt_load_input = Input(shape=(data_dim, 1), dtype='float32', name='wt_load_input')

    # 卷积操作（在经过降维之后，会使用更大的卷积核来组合特征）
    x = Conv1D(30, 8, 3, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(wt_load_input)

    x = Conv1D(30, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(30, 3, 2, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(30, 3, 2, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(30, 3, 2, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(16, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(16, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(16, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(16, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(8, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(8, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(8, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(8, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(4, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(4, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(4, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    x = Conv1D(1, 2, 1, kernel_initializer=kernel_initializer, activation='relu', padding='SAME')(x)

    # 全局池化作为输出层
    x = GlobalAveragePooling1D()(x)

    model = Model(inputs=[wt_load_input], outputs=x)

    return model


# deep WT_CNN
def wt_cnn():
    wt_load_input = Input(shape=(5, data_dim, 1), dtype='float32', name='wt_load_input')

    # 卷积操作（在经过降维之后，会使用更大的卷积核来组合特征）
    x = Conv2D(30, (1, 8), strides=(1, 3), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(wt_load_input)
    x = Conv2D(30, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = Conv2D(30, (3, 3), strides=(1, 2), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(30, (3, 3), strides=(1, 2), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(30, (3, 3), strides=(1, 2), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = Conv2D(16, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(16, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(16, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = Conv2D(8, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(8, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(8, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = Conv2D(4, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(4, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(4, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = Conv2D(1, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    # 全局池化作为输出层
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=[wt_load_input], outputs=x)

    return model


def wt_cnn_avgpool():
    wt_load_input = Input(shape=(5, data_dim, 1), dtype='float32', name='wt_load_input')

    # 卷积操作（在经过降维之后，会使用更大的卷积核来组合特征）
    x = Conv2D(30, (1, 8), strides=(1, 3), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(wt_load_input)
    x = Conv2D(30, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = Conv2D(30, (3, 3), strides=(1, 2), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(30, (3, 3), strides=(1, 2), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(30, (3, 3), strides=(1, 2), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = AvgPool2D(30, (2, 2), padding="SAME")(x)

    x = Conv2D(16, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(16, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(16, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = AvgPool2D(16, (2, 2), padding="SAME")(x)

    x = Conv2D(8, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(8, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(8, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = Conv2D(4, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(4, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(4, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = Conv2D(1, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    # 全局池化作为输出层
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=[wt_load_input], outputs=x)

    return model


def wt_cnn_maxpool():
    wt_load_input = Input(shape=(5, data_dim, 1), dtype='float32', name='wt_load_input')

    # 卷积操作（在经过降维之后，会使用更大的卷积核来组合特征）
    x = Conv2D(30, (1, 8), strides=(1, 3), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(wt_load_input)
    x = Conv2D(30, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = Conv2D(30, (3, 3), strides=(1, 2), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(30, (3, 3), strides=(1, 2), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(30, (3, 3), strides=(1, 2), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = MaxPool2D(30, (2, 2), padding="SAME")(x)

    x = Conv2D(16, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(16, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(16, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = MaxPool2D(16, (2, 2), padding="SAME")(x)

    x = Conv2D(8, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(8, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(8, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = Conv2D(4, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(4, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)
    x = Conv2D(4, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    x = Conv2D(1, (2, 2), strides=(1, 1), kernel_initializer=kernel_initializer, activation='relu',
               padding='SAME')(x)

    # 全局池化作为输出层
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=[wt_load_input], outputs=x)

    return model


data_dim = 4 * 24
kernel_initializer = he_normal(seed=123)
batch_size = 2 * 64
epochs = 600

if __name__ == '__main__':
    # 设置只使用一块GPU并且动态按照所需分配内存
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    __console = sys.stdout  # 用于后期还原

    # 把打印重定向文件
    f = open('outfile_my_code.txt', "a+")
    sys.stdout = f

    # 实验均做5次取平均值
    for t in range(0, 5):
        (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(0,data_dim,level=2,hs_mode='hard')

        wt_load_train = wt_load_train.reshape(-1, data_dim, 1)
        # print(wt_load_train.shape)
        wt_load_test = wt_load_test.reshape(-1, data_dim, 1)


        model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=SimpleCNN(), epoch=epochs, name='None_WT_1DCNN',batch_size=batch_size)

        (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(1,data_dim,level=2,hs_mode='hard')

        wt_load_train = wt_load_train.reshape(-1, data_dim, 1)
        # print(wt_load_train.shape)
        wt_load_test = wt_load_test.reshape(-1, data_dim, 1)


        model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=SimpleCNN(), epoch=epochs, name='WT_1DCNN',batch_size=batch_size)
        #
        # # 选择软硬阈值1DCNN，最后选择硬阈值
        # (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(1, data_dim, level=2,hs_mode='hard')
        #
        # wt_load_train = wt_load_train.reshape(-1, data_dim, 1)
        # # print(wt_load_train.shape)
        # wt_load_test = wt_load_test.reshape(-1, data_dim, 1)
        #
        # model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=SimpleCNN(), epoch=epochs, name='WT_1DCNN_hard',batch_size=batch_size)
        #
        #
        # (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(1, data_dim, level=2,hs_mode='soft')
        # wt_load_train = wt_load_train.reshape(-1, data_dim, 1)
        # # print(wt_load_train.shape)
        # wt_load_test = wt_load_test.reshape(-1, data_dim, 1)
        #
        # model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=SimpleCNN(), epoch=epochs,
        #                      name='WT_1DCNN_soft', batch_size=batch_size)
        #
        # 选择软硬阈值 2DCNN
        # (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(4, data_dim,level=2, hs_mode="soft")
        # wt_load_train = wt_load_train.reshape(-1, 5, data_dim, 1)
        # wt_load_test = wt_load_test.reshape(-1, 5, data_dim, 1)
        #
        # model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=wt_cnn_maxpool(), epoch=epochs,
        #                      name='WT_2DCNN_maxpool_soft', batch_size=batch_size)
        #
        #
        # (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(4, data_dim,level = 2,hs_mode="hard")
        # wt_load_train = wt_load_train.reshape(-1, 5, data_dim, 1)
        # wt_load_test = wt_load_test.reshape(-1, 5, data_dim, 1)
        # model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=wt_cnn_maxpool(), epoch=epochs, name='WT_2DCNN_maxpool_hard', batch_size=batch_size)

        #
        #
        # 选择分解层数
        # for i in range(1,6):
        #     (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(1, data_dim,level=i, hs_mode="hard")
        #     wt_load_train = wt_load_train.reshape(-1, data_dim, 1)
        #     # print(wt_load_train.shape)
        #     wt_load_test = wt_load_test.reshape(-1, data_dim, 1)
        #     model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=SimpleCNN(), epoch=epochs,
        #                          name='WT_1DCNN_level_'+str(i), batch_size=batch_size)
        #
        # 选择层数最后为2
        # for i in range(1,6):
        #     (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(4, data_dim,level=i, hs_mode="hard")
        #     wt_load_train = wt_load_train.reshape(-1, 5, data_dim, 1)
        #     wt_load_test = wt_load_test.reshape(-1, 5, data_dim, 1)
        #
        #     model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=wt_cnn_maxpool(), epoch=epochs,
        #                          name='WT_2DCNN_level_'+str(i), batch_size=batch_size)

        # # 选择小波基函数的实验——最终选择function5 也就是4种小波混杂的
        # for i in range(0, 8):
        #     (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(i, data_dim, level=2, hs_mode='hard')
        #
        #     wt_load_train = wt_load_train.reshape(-1, data_dim, 1)
        #     wt_load_test = wt_load_test.reshape(-1, data_dim, 1)
        #
        #     model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=SimpleCNN(), epoch=epochs,
        #                          name='WT_1DCNN_function_' + str(i), batch_size=batch_size)
        # for i in range(2,6):
        #     (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(i, data_dim, level=2, hs_mode='hard')
        #     #
        #     wt_load_train = wt_load_train.reshape(-1, 5, data_dim, 1)
        #     wt_load_test = wt_load_test.reshape(-1, 5, data_dim, 1)
        #     model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=wt_cnn_maxpool(), epoch=epochs, name='WT_2DCNN_maxpool_function_' + str(i), batch_size=batch_size)


        # # 学习率和BatchSize部分的实验 最终选择是lr = 0.001 Batch_Size = 2*64
        # (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(4, data_dim, level=2, hs_mode='hard')
        # for j in range(0,3):
        #     lr = 0.001/(10**j)
        #     # print(lr)
        #     for i in range(1,5):
        #         batch_size = i*64
        #         # print(str(lr)+"batch_size"+str(batch_size))
        #         model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=wt_cnn_maxpool(), epoch=epochs, name='WT_2DCNN_max_'+str(i)+'_64_lr_'+str(lr), batch_size=batch_size,learning_Rate = lr)

        # for j in range(1,6):
        #     for i in range(42,49):
        #         data_dim = i * 2
        #         (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(4, data_dim,level=2, hs_mode="hard")
        #         wt_load_train = wt_load_train.reshape(-1, 5, data_dim, 1)
        #         wt_load_test = wt_load_test.reshape(-1, 5, data_dim, 1)
        #
        #         model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=wt_cnn_maxpool(), epoch=epochs,
        #                              name=str(j)+'_WT_2DCNN_30min*'+str(i), batch_size=batch_size)
        #
        #
        # for i in range(1,8):
        #     data_dim = i * 4 * 24
        #     (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(4, data_dim,level=2, hs_mode="hard")
        #     wt_load_train = wt_load_train.reshape(-1, 5, data_dim, 1)
        #     wt_load_test = wt_load_test.reshape(-1, 5, data_dim, 1)
        #
        #     model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=wt_cnn_maxpool(), epoch=epochs,
        #                          name='WT_2DCNN_1d*'+str(i), batch_size=batch_size)

        ##### 判断加池化层的效果，最终选择最大池化层

        # (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(5, data_dim, level=2, hs_mode='hard')
        # #
        # wt_load_train = wt_load_train.reshape(-1, 5, data_dim, 1)
        # wt_load_test = wt_load_test.reshape(-1, 5, data_dim, 1)
        # model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=wt_cnn(), epoch=epochs, name='WT_2DCNN_function', batch_size=batch_size)
        #
        #
        # (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(5, data_dim, level=2, hs_mode='hard')
        # #
        # wt_load_train = wt_load_train.reshape(-1, 5, data_dim, 1)
        # wt_load_test = wt_load_test.reshape(-1, 5, data_dim, 1)
        # model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=wt_cnn_avgpool(), epoch=epochs, name='WT_2DCNN_avgpool_function', batch_size=batch_size)
        #
        #
        (wt_load_train, y_train), (wt_load_test, y_test) = mlwt.get_dataset(5, data_dim, level=2, hs_mode='hard')
        #
        wt_load_train = wt_load_train.reshape(-1, 5, data_dim, 1)
        wt_load_test = wt_load_test.reshape(-1, 5, data_dim, 1)
        model = common.train(wt_load_train, y_train, wt_load_test, y_test, model=wt_cnn_maxpool(), epoch=epochs, name='WT_2DCNN_maxpool', batch_size=batch_size)

        # model = load_model('model/WT_2DCNN_end.h5')
        # common.predict_plot(model,wt_load_test,y_test,"WT_2DCNN")



    f.close()
    sys.stdout = __console
