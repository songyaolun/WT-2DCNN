import pandas as pd
import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt

def multi_level_wavelet_transform(data, num, level, hs_mode):
    '''
    对输入的信号采用5种不同的小波来将数据扩展为矩阵形式
    :param num: 确定是生成一维还是多维数据
    :param data:负荷数据，数据格式为[x_1,...,x_m]
    :return:
    [[x_1_1,...,x_1_m],[x_2_1,...,x_2_m],
     [x_3_1,...,x_3_m],[x_4_1,...,x_4_m],
     [x_5_1,...,x_5_m],]
    '''
    # 5中不同类型的小波变换后重构的数据
    rec_data_list = []
    # rec_data_list.append(data)

    if num == 0:
        # 使用原数据不是用小波
        wt_type_list = []
        rec_data_list.append(data)
    elif num == 1:
        wt_type_list = ['dmey']
    # elif num == 4:
    #     wt_type_list = ['sym2', 'sym3', 'sym4', 'sym5']
    #     rec_data_list.append(data)
    elif num == 2:
        wt_type_list = ['db1', 'db2', 'db3', 'db4']
        rec_data_list.append(data)
    elif num == 3:
        wt_type_list = ['sym2', 'sym3', 'sym4', 'sym5']
        rec_data_list.append(data)
    elif num == 4:
        wt_type_list = ['coif1', 'coif2', 'coif3', 'coif4']
        rec_data_list.append(data)
    elif num == 5:
        wt_type_list = ['coif1', 'db1', 'sym2', 'dmey']
        rec_data_list.append(data)


    # elif num == 1:
    #     # 1种类型的小波
    #     wt_type_list = ['haar']
    # elif num == 2:
    #     # 1种类型的小波
    #     wt_type_list = ['db1']
    # elif num == 3:
    #     # 1种类型的小波
    #     wt_type_list = ['sym2']
    # elif num == 4:
    #     # 1种类型的小波
    #     wt_type_list = ['coif1']
    # elif num == 5:
    #     # 1种类型的小波
    #     wt_type_list = ['bior1.1']
    # elif num == 6:
    #     # 1种类型的小波
    #     wt_type_list = ['rbio1.1']
    # elif num == 7:
    #     # 1种类型的小波
    #     wt_type_list = ['dmey']





    # for type in wt_type_list:
    #     # 2-level的小波分解
    #     cA2, cD2, cD1 = pywt.wavedec(data, type, level=2)
    #
    #     # 去掉低频的cD2分量来达到去噪的目的
    #     rec = pywt.waverec([cA2, None, cD1], type)
    #
    #     rec_data_list.append(rec)

    for type in wt_type_list:
        # 2-level的小波分解
        coeffs = pywt.wavedec(data, type, level=level)

        # 噪声方差估计：dealt=median(cA2)/0.6754
        dealt = np.median(coeffs[-1])  # 通过cD1计算dealt

        # 阈值计算
        # 这个是固定阈值计算方法
        threshold = dealt * np.sqrt(2 * np.log(len(data)))

        # 使用软或硬阈值处理小波系数
        for i in range(level):
            coeffs[-(i + 1)] = pywt.threshold(coeffs[-(i + 1)], value=threshold, mode=hs_mode, substitute=0)

        # 对硬阈值处理后的数据重构
        rec = pywt.waverec(coeffs=coeffs, wavelet=type)
        rec_data_list.append(rec)

    return rec_data_list


def get_dataset(num, data_dim ,level =2,hs_mode = "hard"):
    # 加载数据集
    df = pd.read_csv('../data/traffic.csv', parse_dates=['date'], index_col='date')

    # 获取前两个月的数据集作为训练集
    train_data = df[(df['month'] <= 2)]
    df_load_data_train = train_data['demand'].values

    # 小波变换的负荷数据和标签
    wt_load_train = []
    y_train = []

    for i in range(data_dim, len(df_load_data_train)):
        # 小波变换的负荷数据
        wt_data = multi_level_wavelet_transform(df_load_data_train[i - data_dim: i], num,level,hs_mode)
        wt_load_train.append(wt_data)

        # 添加标签
        y_train.append(df_load_data_train[i])

    # -------------------------------------------------

    # 获取第三个月作为测试集
    test_data = df[(df['month'] >2 ) ]

    df_load_data_test = test_data['demand'].values

    wt_load_test = []
    y_test = []

    for i in range(data_dim, len(df_load_data_test)):
        # 小波变换的负荷数据
        wt_data = multi_level_wavelet_transform(df_load_data_test[i - data_dim: i], num,level,hs_mode)
        wt_load_test.append(wt_data)

        # 添加标签
        y_test.append(df_load_data_test[i])

    wt_load_train = np.asarray(wt_load_train)

    y_train = np.asarray(y_train)

    wt_load_test = np.asarray(wt_load_test)

    y_test = np.asarray(y_test)

    return (wt_load_train, y_train), (wt_load_test, y_test)


def predict_mlwt():
    df = pd.read_csv('../data/traffic.csv', parse_dates=['date'], index_col='date')
    time_scale = 24*4*3
    # 获取前两个月的数据集作为训练集
    train_data = df[(df['month'] <= 1)]
    df_load_data_train = train_data['demand'].values
    true_data = df_load_data_train[0:time_scale]

    trues = true_data


    coeffs = pywt.wavedec(true_data, "dmey", level=1)
    wt_data = coeffs[0]
    noise_data = coeffs[1]
    # print(wt_data.size)

    plt.plot(true_data, c='#31859B', label='true')
    plt.legend()
    plt.savefig("true.png")
    plt.show()

    plt.plot(wt_data, c='#789440', label='wt')
    plt.axis('off')  # 去掉坐标轴
    plt.savefig("wt.png")
    plt.show()


    plt.plot(noise_data,c='#865DA6',label='noise')
    plt.axis('off')  # 去掉坐标轴
    plt.savefig("noise.png")
    plt.show()
predict_mlwt()