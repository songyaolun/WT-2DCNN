import os
import datetime
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

model_path = 'model/'
init_lr = 0.001

def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1


def train(X_train, Y_train, X_valid, Y_valid, epoch, model, name ,batch_size,learning_Rate=init_lr):
    # 设置学习率
    lr = learning_Rate
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 在Linux服务器上的路径

    log_dir = os.path.join("./logs/fit/", name, current_time)

    #在Windows平台上的路径
    # log_dir = os.path.join(".\\logs\\fit\\",name, current_time)

    def scheduler(epoch):
        # 每200次更新一次学习率
        interval_epoch = 150
        if epoch % interval_epoch == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr / 10)
            # print("lr changed to {}".format(lr / 10))
        return K.get_value(model.optimizer.lr)

    model.compile(loss='mae',
                  optimizer=keras.optimizers.Adam(lr),
                  metrics=[mean_squared_error, mean_absolute_percentage_error])
    # model.summary()
    # 保存最佳模型
    checkpoint = ModelCheckpoint(filepath=model_path + name + '.h5', monitor='val_mean_absolute_percentage_error',
                                 mode='min',
                                 save_best_only='True', verbose=0)

    # checkpoint = ModelCheckpoint(filepath=model_path +name+'.h5', monitor='val_mean_squared_error',
    #                              mode='min',
    #                              save_best_only='True', verbose=1)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2,write_images=True)

    # 动态设置学习率
    reduce_lr = LearningRateScheduler(scheduler)

    startdate = datetime.datetime.now()  # 获取当前时间
    startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式

    history = model.fit(X_train, Y_train, epochs=epoch, verbose=0, validation_data=(X_valid, Y_valid),batch_size = batch_size,
                        callbacks=[reduce_lr, checkpoint, tensorboard_callback])

    enddate = datetime.datetime.now()  # 获取当前时间
    enddate = enddate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式
    index = np.argmin(history.history['val_mean_absolute_percentage_error'])

    # 输出模型名字，loss,mape,time
    print(name, history.history['val_loss'][index], history.history['val_mean_absolute_percentage_error'][index],
          subtime(startdate, enddate))
    # index = np.argmin(history.history['val_mean_squared_error'])
    # print(history.history['val_loss'][index], history.history['val_mean_squared_error'][index])

    return model


def predict_plot(model, test_x_data, test_y_data,name):
    preds = model.predict(test_x_data)
    trues = test_y_data
    plt.plot(preds, c='blue', label='predict')
    plt.plot(trues, c='black', label='true')
    plt.legend()
    plt.title(name)
    plt.savefig(name+".png")
    plt.show()
