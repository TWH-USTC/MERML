import pandas as pd
import matplotlib as mpl
import numpy as np
from init_func import global_variable_init, data_init
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
import globalvaribale
from joblib import dump
import copy
from preprocess import encoder, k_fold_mask, list_reduce
from joblib import load
import shap
from alepython import ale_plot


def huber(gap):
    sigma = globalvaribale.get_value("SIGMA")
    grad = np.sign(sigma) * sigma * gap / np.sqrt(sigma ** 2 + gap ** 2)
    hess = np.sign(sigma) * (sigma ** 3) / np.power(sigma ** 2 + gap ** 2, 1.5)
    return grad, hess


def huber_object_matrix(pred, data_matrix):
    gap = pred - data_matrix.get_label()
    grad, hess = huber(gap)
    return grad, hess


def f_1(x, K):
    return np.exp(K*x)


def f_2(x, K):
    return 1/(1+K*x)


def f_3(x, K):
    return np.sqrt(1/K*x+1)


def f_0(x, K):
    return (1 - K*x)


def fit_rate_constant(data, func):
    from scipy import optimize
    curves = data[:, -7:]
    time = [0, 2, 5, 9, 15, 22, 30]
    k_list = []
    p_list = []
    curves_pred = []
    for index in range(data.shape[0]):
        element = curves[index, :]
        element_unzero = [conc for conc in element if conc > 0]
        time_unzero = time[0:len(element_unzero)]
        coeff, p = optimize.curve_fit(func, time_unzero, element_unzero)
        curve_pred = [func(time[index], *coeff) if index < len(element_unzero) else 0 for index in range(len(time))]
        curves_pred.append(curve_pred)
        k_list.append(coeff[0])
        p_list.append(p[0][0])
    data = np.c_[data[:, 0:-7], np.array(k_list).reshape(-1, 1)]
    data_pred = np.array(curves_pred)[:, 1:].reshape(1, -1)[0]
    data_true = np.array(curves)[:, 1:].reshape(1, -1)[0]
    print(r2_score(data_true, data_pred), mean_absolute_error(data_true, data_pred),
          np.sqrt(mean_squared_error(data_true, data_pred)))
    return data



class MCB:
    def __init__(self, base_model, train, test, lag, shifts=[1, 2, 3],
                 train_time=[2, 5, 9, 15, 22, 30], test_time=[2, 5, 9, 15, 22, 30], flag=False, mask=[]):
        self.base_model = copy.copy(base_model)
        self.train_matrix = None
        self.model_trained = None
        self.train = train
        self.test = test
        self.lag = lag
        self.shifts = shifts
        self.train_time = train_time  # the sampling point time fo the kinetic profile in training set
        self.test_time = test_time  # the sampling point time fo the kinetic profile in training set
        self.flag = flag  # save the model or not
        self.mask = mask  # filter the feature whose index is not in mask

    @staticmethod
    def train_struct(train, time, lag, shifts):
        number_sample = train.shape[0]
        condition = train[:, 0:-7]
        available_time = [2, 5, 9, 15, 22, 30]
        index = [available_time.index(t) for t in time]
        curve = (train[:, -6:])[:, index]
        mask_plus = [0 for index in range(number_sample)]
        n_col = curve.shape[1]

        for shift in shifts:
            time_temp = time[shift - 1:]
            time_pre = np.c_[np.zeros((number_sample, shift + lag - 1)),
                             np.array([time[0:len(time) - shift]])[mask_plus]]
            curve_pre = np.c_[np.ones((number_sample, shift + lag - 1)), curve[:, 0:n_col - shift]]
            for index in range(len(time_temp)):
                time_value = time_temp[index]
                curve_temp = curve[:, index + shift - 1]
                time_pre_temp = time_pre[:, index + shift - 1:index + lag + shift - 1]
                time_gap_temp = time_value - time_pre_temp
                curve_pre_temp = curve_pre[:, index + shift - 1:index + lag + shift - 1]

                x_temp = np.c_[np.c_[np.c_[condition, time_pre_temp], time_gap_temp], curve_pre_temp]
                y_temp = curve_temp.reshape(-1, 1)
                x = x_temp if index == 0 else np.r_[x, x_temp]
                y = y_temp if index == 0 else np.r_[y, y_temp]
            x_data = x if shifts.index(shift) == 0 else np.r_[x_data, x]
            y_data = y if shifts.index(shift) == 0 else np.r_[y_data, y]

        shift_min = min(shifts)
        if shift_min == 1:
            pass
        else:
            time_temp = time[0:shift_min - 1]
            time_pre = np.zeros((number_sample, lag))
            curve_temp = curve[:, 0:shift_min - 1].reshape(number_sample, -1)
            curve_pre = np.ones((number_sample, lag))
            for index in range(len(time_temp)):
                time_gap_temp = time_temp[index] - time_pre
                x_temp = np.c_[np.c_[np.c_[condition, time_pre], time_gap_temp], curve_pre]
                y_temp = curve_temp[:, index].reshape(-1, 1)
                x = x_temp if index == 0 else np.r_[x, x_temp]
                y = y_temp if index == 0 else np.r_[y, y_temp]
            x_data = np.r_[x_data, x]
            y_data = np.r_[y_data, y]
        # np.savetxt('train.txt', np.c_[np.c_[x_data[:, 0], x_data[:, -3:]], y_data])
        return x_data, y_data

    def train_loop(self):
        x, y = self.train_struct(self.train, self.train_time, self.lag, self.shifts)
        if len(self.mask) != 0:
            x = x[:, self.mask]
        train_matrix = xgb.DMatrix(x, label=y)
        if isinstance(self.base_model, str):
            if self.base_model == 'XGBoost':
                params = copy.copy(globalvaribale.get_value('params'))
                num_boost_round = params['num_boost_round']
                del params['num_boost_round']
                model_train = xgb.train(params, dtrain=train_matrix, obj=huber_object_matrix,
                                        num_boost_round=num_boost_round)
                y_pred = model_train.predict(train_matrix)
                # EP_plot(y*100, y_pred*100)
            else:
                raise ValueError
        else:
            model_train = self.base_model.fit(x, y.ravel())
            y_pred = model_train.predict(x)
        self.model_trained = model_train
        if self.flag:
            dump(model_train, 'Result/MCB')
        return y, y_pred

    def inference(self, data):
        y_train_true, y_train_pred = self.train_loop()
        model_trained = self.model_trained
        y_test_true, y_test_pred, x_data, preds = self.inference_method(model_trained, data, self.lag, self.test_time,
                                                                        self.shifts, self.mask)
        return y_test_true, y_test_pred, x_data, preds, y_train_true, y_train_pred

    @staticmethod
    def inference_method(model, data, lag, test_time, shifts, mask):
        number_sample = data.shape[0]
        condition = data[:, 0:-7]

        available_time = [2, 5, 9, 15, 22, 30]
        index = [available_time.index(t) for t in test_time]
        curve = copy.copy(data[:, -6:])[:, index]
        curve_ori = copy.copy(data[:, -6:])[:, index]
        mask_plus = [0 for index in range(number_sample)]

        pred = [[] for index in range(len(shifts) + 1)]
        x_data = [[] for index in range(len(shifts) + 1)]
        shift_min = min(shifts)
        for index in range(len(test_time)):
            if index <= shift_min - 1:
                time_value = (test_time[0:shift_min])[index]
                time_pre = np.zeros((number_sample, lag))
                curve_pre = np.ones((number_sample, lag))
                time_gap = time_value - time_pre
                x_temp = np.c_[np.c_[np.c_[condition, time_pre], time_gap], curve_pre]
                if len(mask) != 0:
                    x_temp = x_temp[:, mask]
                y_temp = curve[:, index].reshape(number_sample, -1)
                try:
                    pred_temp = model.predict(x_temp)
                except:
                    x_matrix = xgb.DMatrix(x_temp, y_temp)
                    pred_temp = model.predict(x_matrix)
                x_data[-1].append(x_temp)
                pred[-1].append(pred_temp)
                curve[:, index] = pred_temp
            else:
                time_value = test_time[index]
                y_temp = curve_ori[:, index].reshape(-1, 1)
                flag = False
                count = 0
                for shift in shifts:
                    time_pre_ele = [
                        test_time[index - shift - lag + index_t + 1] if index - shift - lag + index_t + 1 >= 0
                        else 0 for index_t in range(lag)]
                    if sum(time_pre_ele) == 0:
                        flag = True
                    time_gap_ele = time_value - np.array(time_pre_ele)
                    time_pre = np.array([time_pre_ele])[mask_plus]
                    time_gap = np.array([time_gap_ele])[mask_plus]
                    curve_pre = (np.c_[np.ones((number_sample, shift + lag - 1)),
                                       curve[:, 0:curve_ori.shape[1] - shift]])[:, index:index + lag]
                    x_temp = np.c_[np.c_[np.c_[condition, time_pre], time_gap], curve_pre]
                    if len(mask) != 0:
                        x_temp = x_temp[:, mask]
                    try:
                        y_pred = model.predict(x_temp)
                    except:
                        x_matrix = xgb.DMatrix(x_temp, label=y_temp)
                        y_pred = model.predict(x_matrix)
                    if shifts.index(shift) == 0:
                        pred_temp = copy.copy(y_pred)
                    else:
                        pred_temp += y_pred
                    count += 1
                    x_data[shifts.index(shift)].append(x_temp)
                    pred[shifts.index(shift)].append(y_pred)
                    if flag:
                        break
                curve[:, index] = pred_temp / count
        return curve_ori.reshape(-1, 1), curve.reshape(-1, 1), x_data, pred

    def evaluate(self):
        test_result = self.inference(self.test)
        return r2_score(test_result[0], test_result[1]), mean_absolute_error(test_result[0], test_result[1]), \
               np.sqrt(mean_squared_error(test_result[0], test_result[1]))


class MT:
    def __init__(self, base_model, train, test, train_time=[2, 5, 9, 15, 22, 30], test_time=[2, 5, 9, 15, 22, 30]):
        self.base_model = copy.copy(base_model)
        # print(base_model.get_params())
        self.train_matrix = None
        self.train = train
        self.test = test
        self.train_time = train_time
        self.test_time = test_time

    @staticmethod
    def data_struct(data, data_time):
        all_time = [2, 5, 9, 15, 22, 30]
        index = [all_time.index(t) for t in data_time]
        time = np.array(data_time).reshape(-1, 1)
        curve = data[:, -6:]
        conversions = curve[:, index].reshape(-1, 1)

        number_sample = data.shape[0]
        index_time = [index % len(time) for index in range(number_sample * len(time))]
        index_sample = [index // len(time) for index in range(number_sample * len(time))]
        data_input = data[index_sample, 0:-7]
        data_time = time[index_time, :]
        x = np.c_[data_input, data_time]
        y = conversions
        return x, y

    def train_loop(self):
        x, y = MT.data_struct(self.train, self.train_time)
        if isinstance(self.base_model, str):
            if self.base_model == 'XGBoost':
                train_matrix = xgb.DMatrix(x, label=y)
                params = copy.copy(globalvaribale.get_value('params'))
                num_boost_round = params['num_boost_round']
                del params['num_boost_round']
                model_train = xgb.train(params, dtrain=train_matrix, obj=huber_object_matrix,
                                        num_boost_round=num_boost_round)
                y_pred = model_train.predict(train_matrix)
            else:
                raise ValueError
        else:
            model_train = self.base_model.fit(x, y.ravel())
            y_pred = model_train.predict(x)
        self.model_trained = model_train
        return y, y_pred

    def inference(self, data):
        y_train_true, y_train_pred = self.train_loop()
        y_test_true, y_test_pred, x = self.inference_method(self.model_trained, data, self.test_time)
        return y_test_true, y_test_pred, x, y_train_true, y_train_pred

    @staticmethod
    def inference_method(model, data, test_time):
        x, y = MT.data_struct(data, test_time)
        try:
            pred = model.predict(x)
        except:
            data_matrix = xgb.DMatrix(x, label=y)
            pred = model.predict(data_matrix)
        return y.reshape(-1, 1), pred.reshape(-1, 1), x

    def evaluate(self):
        test_result = self.inference(self.test)
        return r2_score(test_result[0], test_result[1]), mean_absolute_error(test_result[0], test_result[1]), \
               np.sqrt(mean_squared_error(test_result[0], test_result[1]))


def mkm(train, test, model, f_n=f_2):
    curves_test = test[:, -7:]
    train_temp = fit_rate_constant(train)
    test_temp = fit_rate_constant(test)

    if model == 'XGBoost':
        train_matrix = xgb.DMatrix(train_temp[:, :-1], label=train_temp[:, -1])
        params = copy.copy(globalvaribale.get_value('params'))
        num_boost_round = params['num_boost_round']
        del params['num_boost_round']
        # globalvaribale.set_value("SIGMA", 0.2)
        model_train = xgb.train(params, dtrain=train_matrix,
                                num_boost_round=num_boost_round) #obj=huber_object_matrix,
        test_matrix = xgb.DMatrix(test_temp[:, :-1], label=test_temp[:, -1])
        y_test_pred = model_train.predict(test_matrix)
        # print(y_test_pred)
    else:
        model_trained = model.fit(train_temp[:, :-1], train_temp[:, -1])
        y_test_pred = model_trained.predict(test_temp[:, :-1])

    # error = np.abs(test_temp[:, -1] - y_test_pred)
    # mask = np.argsort(error)[:-4]
    # test_temp = test_temp[mask, :]
    # y_test_pred = y_test_pred[mask]
    # curves_test = test[mask, -7:]

    time = [0, 2, 5, 9, 15, 22, 30]
    curves_pred = []
    for k in range(y_test_pred.shape[0]):
        curve_temp = []
        for index in range(len(time)):
            if f_n(time[index], y_test_pred[k]) >= 1:
                curve_temp.append(1)
            elif 1 > f_n(time[index], y_test_pred[k]) > 0:
                curve_temp.append(f_n(time[index], y_test_pred[k]))
            else:
                curve_temp.append(0)
        curves_pred.append(curve_temp)
    curves_pred = np.array(curves_pred)
    return curves_test[:, 1:].reshape(-1, 1), curves_pred[:, 1:].reshape(-1, 1), test_temp[:, -1], y_test_pred


def evaluation_k_fold(data, basemodel, lag=0, shifts=[1], train_time=[2, 5, 9, 15, 22, 30], k=5):
    mask_cv = k_fold_mask(data.shape[0], k)
    mask_all = [index for index in range(data.shape[0])]
    metrics = []
    for mask_test in mask_cv:
        train_temp = data[list(list_reduce(mask_all, mask_test)), :]
        test_temp = data[mask_test, :]
        if basemodel == 'random':
            y_true = test_temp[:, -6:].reshape(1, -1).tolist()[0]
            np.random.seed(globalvaribale.get_value('SEED'))
            y_pred = np.random.rand(len(y_true))
        else:
            if lag == -1:
                y_true, y_pred = mkm(train_temp, test_temp, basemodel)
            elif lag == 0:
                model = MT(basemodel, train_temp, test_temp, train_time)
                y_true, y_pred, x, y_train_true, y_train_pred = model.inference(test_temp)
                # print(y_pred[0:10, 0].tolist())
            else:
                model = MCB(basemodel, train_temp, test_temp, lag, shifts, train_time)
                y_true, y_pred, x_data, pred, y_train_true, y_train_pred = model.inference(test_temp)
        try:
            metrics.append([mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))])
        except:
            metrics.append([-1, -1])
    return metrics


def baseline_evaluation(fps, models, models_name, lags, shifts, k_fold=5):
    metrics = []
    info = []
    for index_m in range(len(models)):
        for index_f in range(len(fps)):
            for index_l in range(len(lags)):
                for index_s in range(len(shifts)):
                    model = models[index_m]
                    fp = fps[index_f]
                    lag = lags[index_l]
                    shift = shifts[index_s]
                    info_temp = models_name[index_m] + '-' + fp + '-' + str(lag) + '-' + str(index_s + 1)
                    info.append(info_temp)

                    train, test, data = data_init('evaluate')
                    train = encoder(train, fp)
                    metric = evaluation_k_fold(train, model, lag, shift, k=k_fold)
                    metrics.append(metric)
                    print(info_temp + ':', end=' ')
                    print(np.mean(np.array(metric), axis=0))
                    if index_l == 0 and lags[index_l] == 0:
                        break
    return metrics, info


def baseline_generate(type):
    mlp = MLPRegressor()
    knn = KNeighborsRegressor()
    bag = BaggingRegressor()
    svr = SVR()
    kr = KernelRidge()
    dt = DecisionTreeRegressor()
    ada = AdaBoostRegressor()
    rf = RandomForestRegressor()
    et = ExtraTreesRegressor()
    xgbr = xgb.sklearn.XGBRegressor()
    lr = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()

    if type == 'fps_alg':
        fps = ['RDKit', 'Morgan/Circular', '3D-Morse', 'WHIM', 'RDKit_WHIM', 'Morgan_WHIM', 'RDKit_Morse',
               'Morgan_Morse']
        models = [xgbr, svr, kr, knn, bag, dt, mlp, rf, et, ada]  #
        models_name = ['xgb', 'svr', 'kr', 'knn', 'bag', 'dt', 'mlp', 'rf', 'et', 'ada']
        lags = [0, 1]  # lag = 0 if for model MT
        shifts = [[1]]
        metrics, info = baseline_evaluation(fps, models, models_name, lags, shifts)
    elif type == 'lag':
        fps = ['Morgan_WHIM']
        models = [xgbr]
        models_name = ['xgb']
        lags = [0, 1, 2, 3, 4, 5]
        shifts = [[1]]
        metrics, info = baseline_evaluation(fps, models, models_name, lags, shifts)
    elif type == 'shift':
        fps = ['Morgan_WHIM']
        models = [xgbr]
        models_name = ['xgb']
        lags = [1]
        shifts = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
        metrics, info = baseline_evaluation(fps, models, models_name, lags, shifts)
    elif type == 'linear_des_alg':
        fps = ['RDKit', 'Morgan/Circular', '3D-Morse', 'WHIM', 'RDKit_WHIM', 'Morgan_WHIM', 'RDKit_Morse',
               'Morgan_Morse']
        models = [lr, ridge, lasso]  #
        models_name = ['lr', 'ridge', 'lasso']
        lags = [0, 1]  # lag = 0 if for model MT
        shifts = [[1]]
        metrics, info = baseline_evaluation(fps, models, models_name, lags, shifts)
    elif type == 'linear_lag':
        fps = ['WHIM']
        models = [ridge]  #
        models_name = ['ridge']
        lags = [0, 1, 2, 3, 4, 5]  # lag = 0 if for model MT
        shifts = [[1]]
        metrics, info = baseline_evaluation(fps, models, models_name, lags, shifts)
    elif type == 'linear_shift':
        fps = ['WHIM']
        models = [ridge]  #
        models_name = ['ridge']
        lags = [1]  # lag = 0 if for model MT
        shifts = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
        metrics, info = baseline_evaluation(fps, models, models_name, lags, shifts)
    return metrics, info


def hyper_opt(fp, lags, shifts):
    # mcb: Morgan_WHIM-1-[1, 2, 3]
    # mc: Morgan_WHIM-1-[1]
    # mt: Rdkit_WHIM-0-1
    train, test, data = data_init('evaluate')
    train = encoder(train, fp)
    result_opt = []
    info_opt = []
    sigmas = [0.03]
    for num in range(4, 7, 2):
        for depth in range(6, 10, 1):
            for child in range(10, 20, 2):
                for sigma in sigmas:
                    info = str(num * 100) + '-' + str(depth) + '-' + str(child) + '-' + str(sigma)

                    params = {'eta': 0.1, 'max_depth': depth, 'min_child_weight': child,
                              'num_boost_round': 100 * num}

                    globalvaribale.set_value('SIGMA', sigma)
                    globalvaribale.set_value('params', params)

                    metric = evaluation_k_fold(train, 'XGBoost', lags, shifts)
                    print(info + ': ', end='')
                    print(np.mean(np.array(metric), axis=0))
                    info_opt.append(info)
                    result_opt.append(metric)
    return result_opt, info_opt


def model_compare():
    train, test, data = data_init('evaluate')
    train = encoder(train, 'RDKit_Morse')
    metrics = evaluation_k_fold(train, xgb.sklearn.XGBRegressor(), lag=-1)
    m9 = np.mean(metrics, axis=0)
    s9 = np.std(metrics, axis=0)
    print(np.mean(metrics, axis=0))
    print(np.std(metrics, axis=0))

    train, test, data = data_init('evaluate')
    train = encoder(train, 'RDKit_WHIM')
    params = {'eta': 0.1, 'max_depth': 9, 'min_child_weight': 12,
              'num_boost_round': 600}
    globalvaribale.set_value('SIGMA', 0.03)
    globalvaribale.set_value('params', params)
    metrics = evaluation_k_fold(train, 'XGBoost', 0, [1])
    m1 = np.mean(metrics, axis=0)
    s1 = np.std(metrics, axis=0)
    print(np.mean(metrics, axis=0))
    print(np.std(metrics, axis=0))

    train, test, data = data_init('evaluate')
    train = encoder(train, 'Morgan_WHIM')
    params = {'eta': 0.1, 'max_depth': 9, 'min_child_weight': 14,
              'num_boost_round': 600}
    globalvaribale.set_value('SIGMA', 0.03)
    globalvaribale.set_value('params', params)
    metrics = evaluation_k_fold(train, 'XGBoost', 1, [1])
    m3 = np.mean(metrics, axis=0)
    s3 = np.std(metrics, axis=0)
    print(np.mean(metrics, axis=0))
    print(np.std(metrics, axis=0))

    train, test, data = data_init('evaluate')
    train = encoder(train, 'Morgan_WHIM')
    params = {'eta': 0.1, 'max_depth': 8, 'min_child_weight': 18,
              'num_boost_round': 600}
    globalvaribale.set_value('SIGMA', 0.03)
    globalvaribale.set_value('params', params)
    metrics = evaluation_k_fold(train, 'XGBoost', 1, [1, 2, 3])
    m5 = np.mean(metrics, axis=0)
    s5 = np.std(metrics, axis=0)
    print(np.mean(metrics, axis=0))
    print(np.std(metrics, axis=0))

    train, test, data = data_init('evaluate')
    train = encoder(train, 'WHIM')
    metrics = evaluation_k_fold(train, Ridge(), 1)
    m7 = np.mean(metrics, axis=0)
    s7 = np.std(metrics, axis=0)
    print(np.mean(metrics, axis=0))
    print(np.std(metrics, axis=0))

    # train, test, data = data_init('evaluate')
    # metrics = evaluation_k_fold(train, 'random')
    # m9 = np.mean(metrics, axis=0)
    # s9 = np.std(metrics, axis=0)
    # print(np.mean(metrics, axis=0))
    # print(np.std(metrics, axis=0))

    pos = 0
    result = [[m5[pos], m3[pos], m1[pos], m7[pos], m9[pos]], \
              [s5[pos], s3[pos], s1[pos], s7[pos], s9[pos]]]
    return result


def noise_data(data, std=0.05):
    curve = data[:, -6:].reshape(1, -1).tolist()[0]
    noise_func = lambda x: np.random.normal(0, std, 1) + x
    curve_noise = list(map(noise_func, curve))
    data[:, -6:] = np.array(curve_noise).reshape(-1, 6)
    return data


def evaluate_noise():
    metrics = [[], [], []]
    for index in range(0, 11, 1):
        sd = 0.01 * index

        train, test, data = data_init('evaluate')
        train = noise_data(train, std=sd)
        train = encoder(train, 'Morgan_WHIM')
        params = {'eta': 0.1, 'max_depth': 8, 'min_child_weight': 18,
                  'num_boost_round': 600}
        globalvaribale.set_value('SIGMA', 0.03)
        globalvaribale.set_value('params', params)
        metrics_cb = evaluation_k_fold(train, 'XGBoost', 1, [1, 2, 3])

        train, test, data = data_init('evaluate')
        train = noise_data(train, std=sd)
        train = encoder(train, 'Morgan_WHIM')
        params = {'eta': 0.1, 'max_depth': 9, 'min_child_weight': 14,
                  'num_boost_round': 600}
        globalvaribale.set_value('SIGMA', 0.03)
        globalvaribale.set_value('params', params)
        metrics_c = evaluation_k_fold(train, 'XGBoost', 1, [1])

        train, test, data = data_init('evaluate')
        train = noise_data(train, std=sd)
        train = encoder(train, 'RDKit_WHIM')
        params = {'eta': 0.1, 'max_depth': 9, 'min_child_weight': 12,
                  'num_boost_round': 600}
        globalvaribale.set_value('SIGMA', 0.03)
        globalvaribale.set_value('params', params)
        metrics_t = evaluation_k_fold(train, 'XGBoost', 0, [1])

        print([np.mean(metrics_cb, axis=0), np.mean(metrics_c, axis=0), np.mean(metrics_t, axis=0)])
        metrics[0].append(metrics_cb)
        metrics[1].append(metrics_c)
        metrics[2].append(metrics_t)
    return metrics


def time_mask():
    from itertools import combinations
    data_save = []
    for index in range(1, 7, 1):
        mask_temp = list(combinations([2, 5, 9, 15, 22, 30], index))
        data_save.append(mask_temp)
    return data_save


def evaluate_points():
    metrics = [[] for index in range(6)]
    times_list = time_mask()
    for i in range(len(times_list)):
        times = times_list[i]
        for time in times:
            train_time = list(time)
            print(train_time)
            train, test, data = data_init('evaluate')
            train = encoder(train, 'Morgan_WHIM')
            params = {'eta': 0.1, 'max_depth': 8, 'min_child_weight': 18,
                      'num_boost_round': 600}
            globalvaribale.set_value('SIGMA', 0.03)
            globalvaribale.set_value('params', params)
            metrics_cb = evaluation_k_fold(train, 'XGBoost', 1, [1, 2, 3], train_time)
            print(np.mean(metrics_cb, axis=0))

            train, test, data = data_init('evaluate')
            train = encoder(train, 'Morgan_WHIM')
            params = {'eta': 0.1, 'max_depth': 9, 'min_child_weight': 14,
                      'num_boost_round': 600}
            globalvaribale.set_value('SIGMA', 0.03)
            globalvaribale.set_value('params', params)
            metrics_c = evaluation_k_fold(train, 'XGBoost', 1, [1], train_time)
            print(np.mean(metrics_c, axis=0))

            train, test, data = data_init('evaluate')
            train = encoder(train, 'RDKit_WHIM')
            params = {'eta': 0.1, 'max_depth': 9, 'min_child_weight': 12,
                      'num_boost_round': 600}
            globalvaribale.set_value('SIGMA', 0.03)
            globalvaribale.set_value('params', params)
            metrics_t = evaluation_k_fold(train, 'XGBoost', 0, [1], train_time)
            print(np.mean(metrics_t, axis=0))

            metric_temp = []
            metric_temp.append(metrics_cb)
            metric_temp.append(metrics_c)
            metric_temp.append(metrics_t)
            metrics[i].append(metric_temp)
            print('\n')
        print(metrics)
    return metrics


def lr(train, test, lag, shift, train_time=[2, 5, 9, 15, 22, 30]):
    mask = []
    for j in range(5):
        mask_all = np.arange(0, train.shape[0], 1)
        np.random.shuffle(mask_all)
        mask.append(mask_all.tolist())

    metrics = []
    for index in range(1, 12, 1):
        num = 20 * index
        metrics_temp = [[], [], [], []]
        for j in range(5):
            print(index, end='-')
            print(j, end=': ')
            train_temp = train[mask[j][0:num], :]
            if lag == 0:
                params = {'eta': 0.1, 'max_depth': 9, 'min_child_weight': 12,
                          'num_boost_round': 600}
                globalvaribale.set_value('SIGMA', 0.03)
                globalvaribale.set_value('params', params)
                model = MT('XGBoost', train_temp, test, train_time)
                y_true, y_pred, x_data, y_train_true, y_train_pred = model.inference(test)
            else:
                if len(shift) == 1:
                    params = {'eta': 0.1, 'max_depth': 9, 'min_child_weight': 14,
                              'num_boost_round': 600}
                    globalvaribale.set_value('SIGMA', 0.03)
                    globalvaribale.set_value('params', params)
                else:
                    params = {'eta': 0.1, 'max_depth': 8, 'min_child_weight': 18,
                              'num_boost_round': 600}
                    globalvaribale.set_value('SIGMA', 0.03)
                    globalvaribale.set_value('params', params)
                model = MCB('XGBoost', train_temp, test, lag, shift, train_time)
                y_true, y_pred, x_data, pred, y_train_true, y_train_pred = model.inference(test)
            mae_test = mean_absolute_error(y_true, y_pred)
            rmse_test = np.sqrt(mean_squared_error(y_true, y_pred))
            mae_train = mean_absolute_error(y_train_true, y_train_pred)
            rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
            print(mae_test)
            metrics_temp[0].append(mae_test)
            metrics_temp[1].append(rmse_test)
            metrics_temp[2].append(mae_train)
            metrics_temp[3].append(rmse_train)
        metrics.append(metrics_temp)
    metrics = np.array(metrics)

    pos = 0
    result = [[], []]
    for index in range(11):
        metrics_temp = metrics[index, :, :]
        mean = np.mean(metrics_temp, axis=1)
        std = np.std(metrics_temp, axis=1)
        result[0].append([mean[pos], mean[pos] - std[pos], mean[pos] + std[pos]])
        result[1].append([mean[pos + 2], mean[pos + 2] - std[pos + 2], mean[pos + 2] + std[pos + 2]])
    result = np.array(result)
    return result


def lrs_file_generate():
    global_variable_init()

    # train, test, data = data_init('evaluate')
    # train = encoder(train, 'Morgan_WHIM')
    # test = encoder(test, 'Morgan_WHIM')
    # data_1 = lr(train, test, 1, [1, 2, 3])
    # data_2 = lr(train, test, 1, [1, 2, 3], [2, 9, 22])

    # train, test, data = data_init('evaluate')
    # train = encoder(train, 'Morgan_WHIM')
    # test = encoder(test, 'Morgan_WHIM')
    # data_3 = lr(train, test, 1, [1])
    # data_4 = lr(train, test, 1, [1], [2, 9, 22])

    # train, test, data = data_init('evaluate')
    # train = encoder(train, 'RDKit_WHIM')
    # test = encoder(test, 'RDKit_WHIM')
    # data_5 = lr(train, test, 0, [1])
    # data_6 = lr(train, test, 0, [1], [2, 9, 22])
    #
    train, test, data = data_init('evaluate')
    train = noise_data(train, 0.03)
    train = encoder(train, 'Morgan_WHIM')
    test = encoder(test, 'Morgan_WHIM')
    data_7 = lr(train, test, 1, [1, 2, 3])
    data_8 = lr(train, test, 1, [1, 2, 3], [2, 9, 22])

    train, test, data = data_init('evaluate')
    train = noise_data(train, 0.03)
    train = encoder(train, 'Morgan_WHIM')
    test = encoder(test, 'Morgan_WHIM')
    data_9 = lr(train, test, 1, [1])
    data_10 = lr(train, test, 1, [1], [2, 9, 22])

    train, test, data = data_init('evaluate')
    train = noise_data(train, 0.03)
    train = encoder(train, 'RDKit_WHIM')
    test = encoder(test, 'RDKit_WHIM')
    data_11 = lr(train, test, 0, [1])
    data_12 = lr(train, test, 0, [1], [2, 9, 22])
    #
    # np.save('Result/lrs_mcb_all-5.npy', data_1)
    # np.save('Result/lrs_mcb_part-5.npy', data_2)
    # np.save('Result/lrs_mc_all-5.npy', data_3)
    # np.save('Result/lrs_mc_part-5.npy', data_4)
    # np.save('Result/lrs_mt_all-5.npy', data_5)
    # np.save('Result/lrs_mt_part-5.npy', data_6)
    np.save('Result/lrs_mcb_noise_all-5.npy', data_7)
    np.save('Result/lrs_mcb_noise_part-5.npy', data_8)
    np.save('Result/lrs_mc_noise_all-5.npy', data_9)
    np.save('Result/lrs_mc_noise_part-5.npy', data_10)
    np.save('Result/lrs_mt_noise_all-5.npy', data_11)
    np.save('Result/lrs_mt_noise_part-5.npy', data_12)


def shap_mcb_file_generate(data, num_dispaly):
    model_trained = load("Result/MCB")
    x_shap = MCB.train_struct(data, [2, 5, 9, 15, 22, 30], 1, [1, 2, 3])[0]
    explainer = shap.TreeExplainer(model_trained)

    shap_values = explainer.shap_values(x_shap)

    features = np.loadtxt('title.txt', dtype=str).tolist()
    title = [25, 81, 92, 100, 127, 146, 185, 188, 192, 208, 217, 246, 286, 316, 323, 380, 408, 457, 516, 540, 562, 651,
             673, 695, 696, 707, 716, 754, 777, 839, 849, 876, 890, 919, 932, 936, 1005, 1036, 1058, 1088, 1091, 1108,
             1149, 1153, 1179, 1181, 1196, 1217, 1270, 1275, 1283, 1335, 1355, 1386, 1448, 1479, 1537, 1576, 1593, 1622,
             1657, 1684, 1716, 1739, 1755, 1758, 1769, 1810, 1821, 1824, 1849, 1850, 1918, 1933, 1937, 1964, 2034]
    features_morgan = ['M' + str(t) for t in title]
    # features = features[0:-7] + features_morgan + features[-7:]
    features = features_morgan + features[-7:]
    print(x_shap.shape)
    print(features)
    shap.summary_plot(shap_values, x_shap, max_display=num_dispaly, feature_names=features)

    # some files (data_color, x_pos, y_pos, feature_index, feature_names in directory Result) were extracted from
    # the shap package (file named: _beeswarm; line: 612-686; variable: data_color, x_pos, y_pos, feature_order
    # (for feature_index and feature_names was generated according to this variable))

    data_force = []
    index_0 = 60  # the index of the example reaction in the dataset
    for index in range(6):
        index = index_0 + index * (data.shape[0])
        for i in range(len(features)):
            features[i] = features[i] + '=' + str(np.round(x_shap[index, i], 2))
        shap.force_plot(explainer.expected_value, shap_values[index], matplotlib=True, show=True, figsize=(10, 3),
                        features=features)
        # the file was extracted from the shap package (file named: _force_mtplotlib; line: 428; variable: data_force)
        data_extract = np.load('Result/data_force.npy', allow_pickle=True)
        data_force.append(data_extract)

    np.save('Result/force_data_example_morgan.npy', data_force)


def feature_contribution(mask):
    index = 0
    metric = []
    while ((index + 1) * 5 <= len(mask)):
        mask_attr = mask[0:(index + 1) * 5]
        metric_attr = []
        for seed in range(10):
            globalvaribale.set_value('SEED', seed)
            train, test, data = data_init('evaluate')
            train = encoder(train, 'Morgan_WHIM')
            test = encoder(test, 'Morgan_WHIM')
            params = {'eta': 0.1, 'max_depth': 8, 'min_child_weight': 18,
                      'num_boost_round': 600}
            globalvaribale.set_value('SIGMA', 0.03)
            globalvaribale.set_value('params', params)
            model = MCB('XGBoost', train, test, 1, [1, 2, 3], [2, 5, 9, 15, 22, 30], [2, 5, 9, 15, 22, 30], False,
                        mask_attr)
            y_true, y_pred, x_data, pred, y_train_true, y_train_pred = model.inference(test)
            mae = mean_absolute_error(y_true, y_pred)
            print(mae)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            metric_attr.append([mae, rmse])
        metric.append(metric_attr)
        index += 1
    return metric


class ALE:
    def predict(data):
        model_trained = model_trained = load("Result/MCB")
        func_pred = lambda x: model_trained.predict(xgb.DMatrix(np.array(x), label=np.zeros((x.shape[0], 1))))
        y_pred = func_pred(data)
        return y_pred


def divide_data_for_application_scope_evaluation(data, flag='reactant type'):
    all_index = []
    if flag == 'reactant concentration':
        concentration_ori = data[:, 1:4]
        concentration_com = lambda x: str(x[0]) + '-' + str(x[1]) + '-' + str(x[2])
        concentration = list(map(concentration_com, concentration_ori))
        concentration_unique = np.unique(np.array(concentration)).tolist()
        masks = [[] for index in range(len(concentration_unique))]
        for index in range(data.shape[0]):
            all_index.append(index)
            element = data[index, :]
            conc_element = concentration_com(element[1:4])
            pos = concentration_unique.index(conc_element)
            masks[pos].append(index)
        print(concentration_unique)
    elif flag == 'reactant type':
        reactant_unique = np.unique(data[:, 0]).tolist()
        masks = [[] for index in range(len(reactant_unique))]
        for index in range(data.shape[0]):
            all_index.append(index)
            element = data[index, :]
            pos = reactant_unique.index(element[0])
            masks[pos].append(index)
        print(reactant_unique)
    return all_index, masks


def application_scope_evaluation(flag='reactant type'):
    global_variable_init()
    train, test, data = data_init('evaluate')
    data_ori = copy.copy(data)
    all_index, masks = divide_data_for_application_scope_evaluation(data_ori, flag)
    data =  encoder(data, 'Morgan_WHIM')
    metrics = []
    for index in range(len(masks)):
        test_mask = masks[index]
        train_mask = list_reduce(all_index, test_mask)
        train = data[train_mask, :]
        test = data[test_mask, :]
        params = {'eta': 0.1, 'max_depth': 8, 'min_child_weight': 18,
                  'num_boost_round': 600}
        globalvaribale.set_value('SIGMA', 0.03)
        globalvaribale.set_value('params', params)
        model = MCB('XGBoost', train, test, 1, [1, 2, 3], [2, 5, 9, 15, 22, 30], [2, 5, 9, 15, 22, 30], False)
        y_true, y_pred, x_data, pred, y_train_true, y_train_pred = model.inference(test)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics.append([r2, mae, rmse])
        print(r2)
    np.save('metrics_application_scope_'+flag+'.npy', metrics)


if __name__ == '__main__':
    '''Obtain the optimal combination of des and alg: Morgan_WHIM+xgb for MCB and Rdkit_WHIM+xgb for MT'''
    # global_variable_init()
    # metrics = []
    # for index in range(5):
    #     globalvaribale.set_value('SEED', index)
    #     metric, info = baseline_generate('fps_alg')
    #     metrics.append(metric)
    # np.save('Result/metrics_base_dess_alg.npy', metrics)
    # np.save('Result/info_base_dess_alg.npy', info)
    # metrics = np.load('Result/metrics_base_dess_alg.npy', allow_pickle=True)
    # info = np.load('Result/info_base_dess_alg.npy', allow_pickle=True)
    # metric_average = np.mean(np.mean(metrics, axis=0), axis=1)
    # mask_sort = np.argsort(metric_average[:, 0])
    # info = info[mask_sort]
    # metric_average = metric_average[mask_sort, :]
    # print(info[0])
    # print(metric_average[0, :])
    # print('\n')


    '''obtain the optimal lags'''
    # global_variable_init()
    # metrics = []
    # for index in range(5):
    #     globalvaribale.set_value('SEED', index)
    #     metric, info = baseline_generate('lag')
    #     metrics.append(metric)
    # np.save('Result/metrics_base_lags.npy', metrics)
    # np.save('Result/info_base_lags.npy', info)
    # metrics = np.load('Result/metrics_base_lags.npy', allow_pickle=True)
    # info = np.load('Result/info_base_lags.npy', allow_pickle=True)
    # metric_average = np.mean(np.mean(metrics, axis=0), axis=1)
    # mask_sort = np.argsort(metric_average[:, 0])
    # info = info[mask_sort]
    # metric_average = metric_average[mask_sort, :]
    # print(info[0])
    # print(metric_average[0, :])
    # print('\n')

    '''obtain the optimal shifts'''
    # metrics = []
    # for index in range(5):
    #     globalvaribale.set_value('SEED', index)
    #     metric, info = baseline_generate('shift')
    #     metrics.append(metric)
    # np.save('Result/metrics_base_shifts.npy', metrics)
    # np.save('Result/info_base_shifts.npy', info)
    # metrics = np.load('Result/metrics_base_shifts.npy', allow_pickle=True)
    # info = np.load('Result/info_base_shifts.npy', allow_pickle=True)
    # metric_average = np.mean(np.mean(metrics, axis=0), axis=1)
    # mask_sort = np.argsort(metric_average[:, 0])
    # info = info[mask_sort]
    # metric_average = metric_average[mask_sort, :]
    # print(info[0])
    # print(metric_average[0, :])
    # print('\n')

    '''optimal hyper-parameters of XGBoost tree in model MCB'''
    # global_variable_init()
    # result_opt, info_opt = hyper_opt('Morgan_WHIM', 1, [1, 2, 3])
    # np.save('Result/result_opt_mcbs.npy', result_opt)
    # np.save('Result/info_opt_mcbs.npy', info_opt)
    # metrics = np.load('Result/result_opt_mcbs.npy', allow_pickle=True)
    # info = np.load('Result/info_opt_mcbs.npy', allow_pickle=True)
    # metric_average = np.mean(metrics, axis=1)
    # mask_sort = np.argsort(metric_average[:, 0])
    # info = info[mask_sort]
    # metric_average = metric_average[mask_sort, :]
    # print(info[0])
    # print(metric_average[0, :])
    # print('\n')

    '''optimal hyper-parameters of XGBoost tree in model MC'''
    # global_variable_init()
    # result_opt, info_opt = hyper_opt('Morgan_WHIM', 1, [1])
    # np.save('Result/result_opt_mcs.npy', result_opt)
    # np.save('Result/info_opt_mcs.npy', info_opt)
    # metrics = np.load('Result/result_opt_mcs.npy', allow_pickle=True)
    # info = np.load('Result/info_opt_mcs.npy', allow_pickle=True)
    # metric_average = np.mean(metrics, axis=1)
    # mask_sort = np.argsort(metric_average[:, 0])
    # info = info[mask_sort]
    # metric_average = metric_average[mask_sort, :]
    # print(info[0])
    # print(metric_average[0, :])
    # print('\n')

    '''optimal hyper-parameters of XGBoost tree in model MT'''
    # global_variable_init()
    # result_opt, info_opt = hyper_opt('RDKit_WHIM', 0, [])
    # np.save('Result/result_opt_mts.npy', result_opt)
    # np.save('Result/info_opt_mts.npy', info_opt)
    # metrics = np.load('Result/result_opt_mts.npy', allow_pickle=True)
    # info = np.load('Result/info_opt_mts.npy', allow_pickle=True)
    # metric_average = np.mean(metrics, axis=1)
    # mask_sort = np.argsort(metric_average[:, 0])
    # info = info[mask_sort]
    # metric_average = metric_average[mask_sort, :]
    # print(info[0])
    # print(metric_average[0, :])
    # print('\n')

    '''linear model des-alg: ridge-WHIM-1-1 and lr-3D-Morse-0-1'''
    # global_variable_init()
    # metrics = []
    # for index in range(5):
    #     globalvaribale.set_value('SEED', index)
    #     metric, info = baseline_generate('linear_des_alg')
    #     metrics.append(metric)
    # np.save('Result/metrics_base_linear_des_alg.npy', metrics)
    # np.save('Result/info_base_linear_des_alg.npy', info)
    # data = np.load('Result/metrics_base_linear_des_alg.npy', allow_pickle=True)
    # info = np.load('Result/info_base_linear_des_alg.npy', allow_pickle=True)
    # metric_average = np.mean(np.mean(data, axis=0), axis=1)
    # mask_sort = np.argsort(metric_average[:, 0])
    # info = info[mask_sort]
    # metric_average = metric_average[mask_sort, :]
    # print(info[0])
    # print(metric_average[0, :])
    # print('\n')

    '''linear model lags: lag = 1'''
    # global_variable_init()
    # metrics = []
    # for index in range(5):
    #     globalvaribale.set_value('SEED', index)
    #     metric, info = baseline_generate('linear_lag')
    #     metrics.append(metric)
    # np.save('Result/metrics_base_linear_lag.npy', metrics)
    # np.save('Result/info_base_linear_lag.npy', info)
    # data = np.load('Result/metrics_base_linear_lag.npy', allow_pickle=True)
    # info = np.load('Result/info_base_linear_lag.npy', allow_pickle=True)
    # metric_average = np.mean(np.mean(data, axis=0), axis=1)
    # mask_sort = np.argsort(metric_average[:, 0])
    # info = info[mask_sort]
    # metric_average = metric_average[mask_sort, :]
    # print(info[0])
    # print(metric_average[0, :])
    # print('\n')

    '''linear model shift: shift = 1'''
    # global_variable_init()
    # metrics = []
    # for index in range(5):
    #     globalvaribale.set_value('SEED', index)
    #     metric, info = baseline_generate('linear_shift')
    #     metrics.append(metric)
    # np.save('Result/metrics_base_linear_shift.npy', metrics)
    # np.save('Result/info_base_linear_shift.npy', info)
    # data = np.load('Result/metrics_base_linear_shift.npy', allow_pickle=True)
    # info = np.load('Result/info_base_linear_shift.npy', allow_pickle=True)
    # metric_average = np.mean(np.mean(data, axis=0), axis=1)
    # mask_sort = np.argsort(metric_average[:, 0])
    # info = info[mask_sort]
    # metric_average = metric_average[mask_sort, :]
    # print(info[0])
    # print(metric_average[0, :])
    # print('\n')

    '''mkm model'''
    # global_variable_init()
    # train, test, data = data_init('evaluate')
    #
    # fps = ['RDKit', 'Morgan/Circular', '3D-Morse', 'WHIM', 'RDKit_WHIM', 'Morgan_WHIM', 'RDKit_Morse',
    #        'Morgan_Morse']
    #
    # mlp = MLPRegressor()
    # knn = KNeighborsRegressor()
    # bag = BaggingRegressor()
    # svr = SVR()
    # kr = KernelRidge()
    # dt = DecisionTreeRegressor()
    # ada = AdaBoostRegressor()
    # rf = RandomForestRegressor()
    # et = ExtraTreesRegressor()
    # xgbr = xgb.sklearn.XGBRegressor()
    # lr = LinearRegression()
    # ridge = Ridge()
    # lasso = Lasso()
    # models = [mlp, knn, bag, svr, kr, dt, ada, rf, et, xgbr, lr, ridge, lasso]
    # model_names = ['mlp', 'knn', 'bag', 'svr', 'kr', 'dt', 'ada', 'rf', 'et', 'xgbr', 'lr', 'ridge', 'lasso']
    #
    # metrics = []
    # infos = []
    # for fp in fps:
    #     train_temp = encoder(train, fp)
    #     for model in models:
    #         metric = evaluation_k_fold(train_temp, model, -1)
    #         metrics.append(metric)
    #         info = model_names[models.index(model)] + '-' + fp
    #         infos.append(info)
    #         print(info, end=': ')
    #         print(metric)
    # np.save('mkm_metrics_f2.npy', metrics)
    # np.save('Temp/mkm_info_f2.npy', infos)

    # data = np.load('mkm_metrics.npy', allow_pickle=True)
    # info = np.load('mkm_info.npy', allow_pickle=True)
    # data = np.mean(data, axis=1)
    # mask = np.argsort(data[:, 0])
    # data = data[mask, :]
    # info = info[mask]
    # print(data)
    # print(info)

    # global_variable_init()
    # result_opt, info_opt = hyper_opt('RDKit_Morse', -1, [1])
    # np.save('Result/result_opt_mkm.npy', result_opt)
    # np.save('Result/info_opt_mkm.npy', info_opt)
    # data = np.load('Result/result_opt_mkm.npy', allow_pickle=True)
    # info = np.load('Result/info_opt_mkm.npy', allow_pickle=True)
    # data = np.mean(data, axis=1)
    # mask = np.argsort(data[:, 0])
    # data = data[mask, :]
    # info = info[mask]
    # print(data)
    # print(info)

    # global_variable_init()
    # train, test, data = data_init('evaluate')
    # train = encoder(train, 'RDKit_Morse')
    # test = encoder(test, 'RDKit_Morse')
    # true, pred, k_true, k_pred = mkm(train, test, xgb.sklearn.XGBRegressor())
    # print(mean_absolute_error(true, pred))

    # import matplotlib.pyplot as plt
    # mpl.rcParams["mathtext.fontset"] = 'custom'
    # mpl.rcParams["mathtext.bf"] = "Arial:bold"
    # mpl.rcParams["mathtext.default"] = 'regular'
    # plt.figure()
    # # mpl.rcParams["markers.fillstyle"] = 'none'
    # left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    # rect = [left, bottom, width, height]
    # ax = plt.axes(rect)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.set_xlim(-2, 1.25)
    # ax.set_xticks([-1.50,  -1.00,  -0.50,  0.00, 0.5, 1.0])
    # ax.set_xticklabels(['-1.50',  '-1.00', '-0.50',  '0.00', '0.50', '1.00'],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # # ax.set_xlim(-1.75, 0.25)
    # # ax.set_xticks([-1.50, -1.00, -0.50, 0.00])
    # # ax.set_xticklabels(['-1.50', '-1.00', '-0.50', '0.00'],
    # #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylim(-1.75, 0.25)
    # ax.set_yticks([-1.50,  -1.00,  -0.50,  0.00])
    # ax.set_yticklabels(['-1.50',  '-1.00', '-0.50',  '0.00'],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylabel('Predicted rate constant', labelpad=5, fontsize=22, fontweight='500',
    #               fontfamily='Arial')
    # ax.set_xlabel('Measured  rate constant', labelpad=5, fontsize=22,
    #               fontfamily='Arial')
    # ax.tick_params(length=4, width=1.5, which='major')
    # plt.scatter(k_true, k_pred, marker='o', s=125, lw=0.2, alpha=0.6, c='#B7E1FC', edgecolor='black', zorder=1)  # #D2D2FF
    # plt.plot([-1.75, 0.25], [-1.75, 0.25], c='#C26275', linestyle='dashed', zorder=0)
    # plt.plot([-0.50, -0.375], [-1, -1], c='#C26275', linestyle='dashed', zorder=0)
    # ax.text(-0.3, -1, s='y=x', fontdict={'family': 'Arial', 'size': '18', 'weight': '500', 'va':'center_baseline'})
    # ax.text(x=-0.5, y=-1.2, s=r'$R^2$' + ' = ' + str(np.round(r2_score(k_true, k_pred), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500', 'va':'center_baseline'})
    # ax.text(x=-0.5, y=-1.40, s='MAE = ' + str(np.round(mean_absolute_error(k_true, k_pred), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500', 'va':'center_baseline'})
    # ax.text(x=-0.5, y=-1.6, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(k_true, k_pred)), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500', 'va':'center_baseline'})
    # plt.savefig('Figures/s8a-1.svg', dpi=600)
    # plt.show()


    '''model compare'''
    # global_variable_init()
    # result = model_compare()
    # np.save('Result/metrics_model_compare.npy', result)

    '''evaluation of noise data'''
    # global_variable_init()
    # metrics = evaluate_noise()
    # print(metrics)
    # np.save('Result/metrics_noise.npy', metrics)

    '''evaluation of part points'''
    # global_variable_init()
    # metrics = evaluate_points()
    # np.save('Result/metrics_points.npy', metrics)

    '''lrs generate'''
    # lrs_file_generate()

    '''model save and E-P plot data generate'''
    # global_variable_init()
    # train, test, data = data_init('evaluate')
    # train = encoder(train, 'Morgan_WHIM')
    # test = encoder(test, 'Morgan_WHIM')
    # params = {'eta': 0.1, 'max_depth': 8, 'min_child_weight': 18,
    #           'num_boost_round': 600}
    # globalvaribale.set_value('SIGMA', 0.03)
    # globalvaribale.set_value('params', params)
    # model = MCB('XGBoost', train, test, 1, [1, 2, 3], [2, 5, 9, 15, 22, 30], [2, 5, 9, 15, 22, 30], True)
    # y_true, y_pred, x_data, pred, y_train_true, y_train_pred = model.inference(test)
    # np.save('Result/result_test.npy', [y_true, y_pred])

    '''generate the files which are needed by the model interpretation'''
    # global_variable_init()
    # train, test, data = data_init('evaluate')
    # data = encoder(data, 'Morgan_WHIM')
    # shap_mcb_file_generate(data, 50)

    '''generate the files needed by plot of the model error with different numbers of features'''
    # mask = np.load('Result/features_index.npy', allow_pickle=True).tolist()
    # mask.reverse()
    # global_variable_init()
    # metric = feature_contribution(mask)
    # np.save('Result/metric_contribution.npy', metric)

    '''calculate the correlation between the features and output using ALE'''
    # global_variable_init()
    # train, test, data = data_init('evaluate')
    # data = encoder(data, 'Morgan_WHIM')
    # x, y = MCB.train_struct(data, [2, 5, 9, 15, 22, 30], 1, [1, 2, 3])
    # x_test = pd.DataFrame(data=x).astype(np.float64)
    # mpl.rc("figure", figsize=(9, 6))
    # index_list = [11, 12, 30, -7, -6, -5, -4, -3, -2, -1]
    # index_list = [-1]
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for index in index_list:
    #     ax = ale_plot(
    #         ALE,
    #         x_test,
    #         x_test.columns[[index]],
    #         bins=8,
    #         )
    #     plt.savefig('Figures/s15-10.svg', dpi=600)

    '''calculate the error caused by introducing different levels of Gaussian noise'''
    # result = []
    # global_variable_init()
    # for index in range(100):
    #     globalvaribale.set_value('SEED', index)
    #     train, test, data = data_init('evaluate')
    #     data = test
    #     y_true = data[:, -6:].reshape(1, -1)
    #     data = noise_data(data, 0.05)
    #     y_pred = data[:, -6:].reshape(1, -1)
    #     result.append((mean_absolute_error(y_true, y_pred)))
    # result = np.array(result).reshape(1, -1)
    # print(np.mean(result, axis=1))
    # print(np.std(result, axis=1))

    '''Figure 2'''
    # from plot import dynamic, autoregression
    # global_variable_init()
    # train, test, data = data_init('evaluate')
    # dynamic(data)
    # autoregression(data)

    '''Figure 4'''
    # from plot import split_plots, lr_plot, plot_model_compare, plot_model_noise, plot_model_points
    # split_plots()
    # plot_model_compare()
    # plot_model_noise()
    # plot_model_points()
    # lr_plot(True)

    '''Figure 5 and S3'''
    # from plot import match_curves
    # match_curves()

    '''Figure 6'''
    # from plot import shap_rewrite, force_plot
    # shap_rewrite()
    # force_plot()

    '''Figure S26'''
    # from plot import baseline_lags, baseline_des_alg, baseline_multis
    # baseline_multis()
    # baseline_lags()
    # baseline_des_alg()

    '''Figure S7'''
    # from plot import lr_plot
    # lr_plot()
    # lr_plot(True)

    '''Figure S9'''
    # from plot import feature_contirbution_plot
    # feature_contirbution_plot()

    '''Fig S11, S12'''
    # global_variable_init()
    # train, test, data = data_init('evaluate')
    # train = encoder(train, 'Morgan/Circular')
    # test = encoder(test, 'Morgan/Circular')
    # params = {'eta': 0.1, 'max_depth': 8, 'min_child_weight': 18,
    #           'num_boost_round': 600}
    # globalvaribale.set_value('SIGMA', 0.03)
    # globalvaribale.set_value('params', params)
    # model = MCB('XGBoost', train, test, 1, [1, 2, 3], [2, 5, 9, 15, 22, 30], [2, 5, 9, 15, 22, 30], True)
    # y_true, y_pred, x_data, pred, y_train_true, y_train_pred = model.inference(test)
    # np.save('Result/result_test.npy', [y_true, y_pred])
    # from plot import EP_plot
    # EP_plot(y_true*100, y_pred*100)

    # global_variable_init()
    # train, test, data = data_init('evaluate')
    # data = encoder(train, 'Morgan/Circular')
    # shap_mcb_file_generate(data, 15)
    # #
    # from plot import shap_rewrite, force_plot
    # shap_rewrite()

    # smis = ['C1=CC=C(C=C1)O', 'C1=CC(=CC=C1O)Cl', 'C1=CC(=CC=C1[N+](=O)[O-])O', 'C1=CC(=CC=C1C(=O)O)O', 'C1=CC(=CC=C1O)O',
    #        'C1=CC(=CC=C1C=O)O', 'CC(=O)C1=CC=C(C=C1)O', 'COC1=CC=C(C=C1)O', 'COC(=O)C1=CC=C(C=C1)O', 'C1=CC(=CC=C1CO)O',
    #        'CC(=O)NC1=CC=C(C=C1)O', 'CC1=CC=C(C=C1)O']
    # bit = [650, 315, 216, 379, 80, 24, 695, 1447]
    # from rdkit.Chem import Draw
    # from rdkit.Chem.Pharm2D import Generate
    # from rdkit import Chem
    # from PIL import Image
    # from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    # from rdkit.Chem.Draw import IPythonConsole
    # from IPython.display import SVG
    # import cairosvg
    # import tempfile
    # drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    # # drawOptions.prepareMolsBeforeDrawing = False
    # for smi in smis:
    #     bi = {}
    #     fp = GetMorganFingerprintAsBitVect(Chem.AddHs(Chem.MolFromSmiles(smi)), radius=2, bitInfo=bi, nBits=2048)
    #     print([x for x in fp.GetOnBits()])
    #     tpls = [(Chem.AddHs(Chem.MolFromSmiles(smi)), x, bi) for x in fp.GetOnBits() if x in bit]
    #     if len(tpls) >= 1:
    #         if len(tpls) >= 2:
    #             for j in range(len(tpls)):
    #                 img = Draw.DrawMorganBits([tpls[j]], molsPerRow=1, useSVG=True,
    #                                           legends=[[str(x + 1) for x in fp.GetOnBits() if x in bit][j]],
    #                                           drawOptions=drawOptions,
    #                                           subImgSize=(200, 200))
    #                 with open('M-' + str(smis.index(smi)) + '-'+str(j) + '.svg', 'w', encoding='utf-8') as f1:
    #                     f1.write(img.data)
    #                     f1.close()
    #
    #         else:
    #             img = Draw.DrawMorganBits(tpls, molsPerRow=1, legends=[str(x + 1) for x in fp.GetOnBits() if x in bit],
    #                                       drawOptions=drawOptions, useSVG=True,
    #                                       subImgSize=(200, 200))
    #             with open('M-'+str(smis.index(smi))+'-0'+'.svg', 'w', encoding='utf-8') as f1:
    #                 f1.write(img.data)
    #                 f1.close()

    # smis = ['C1=CC=C(C=C1)O', 'C1=CC(=CC=C1O)Cl', 'C1=CC(=CC=C1[N+](=O)[O-])O', 'C1=CC(=CC=C1C(=O)O)O', 'C1=CC(=CC=C1O)O',
    #        'C1=CC(=CC=C1C=O)O', 'CC(=O)C1=CC=C(C=C1)O', 'COC1=CC=C(C=C1)O', 'COC(=O)C1=CC=C(C=C1)O', 'C1=CC(=CC=C1CO)O',
    #        'CC(=O)NC1=CC=C(C=C1)O', 'CC1=CC=C(C=C1)O']
    # from rdkit.Chem import Draw
    # from rdkit.Chem.Pharm2D import Generate
    # from rdkit import Chem
    # from PIL import Image
    # from rdkit.Chem import AllChem
    # from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    # from rdkit.Chem.Draw import IPythonConsole
    # from IPython.display import SVG
    # import cairosvg
    # import tempfile
    # drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    # # drawOptions.prepareMolsBeforeDrawing = False
    # for smi in smis:
    #     bi = {}
    #     fp = GetMorganFingerprintAsBitVect(Chem.AddHs(Chem.MolFromSmiles(smi)), radius=2, bitInfo=bi, nBits=2048)
    #     tpls = [(Chem.AddHs(Chem.MolFromSmiles(smi)), x, bi) for x in fp.GetOnBits() ]
    #     img = Draw.DrawMorganBits(tpls, molsPerRow=5, legends=[str(x + 1) for x in fp.GetOnBits()],
    #                                           drawOptions=drawOptions, useSVG=True,
    #                                           subImgSize=(200, 200))
    #     with open('M-' + str(smis.index(smi)) + '.svg', 'w', encoding='utf-8') as f1:
    #         f1.write(img.data)
    #         f1.close()


    '''Table S8 and S9'''
    # data = np.load('Result/metrics_application_scope_reactant type.npy')
    # print(data)
    # application_scope_evaluation()




