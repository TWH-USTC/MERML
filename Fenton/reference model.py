import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from symfit import Variable, Parameter, Fit, D, ODEModel, variables
import seaborn as sns
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.special import lambertw
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sympy as sp
mpl.rcParams["mathtext.fontset"] = 'custom'
mpl.rcParams["mathtext.bf"] = "Arial:bold"
mpl.rcParams["mathtext.default"] = 'regular'


def func(s0, h0, f0, time, conv):
    k1 = 70.0
    k3 = 3.3 * 1e7
    k7 = 3.2 * 1e8

    s, h, f, t = variables('s, h, f, t')
    k = Parameter('k', k7*0.1)

    model_dict = {
        D(s, t): (-1 * k * k1 * f * h * s) / (k * s + k3 * h + k7 * f),
        D(h, t): (-1 * k3 * k1 * f * h * h)/(k*s + k3*h + k7*f),
        D(f, t): (-1 * k7 * k1 * f * h * f) / (k * s + k3 * h + k7 * f),
    }

    ode_model = ODEModel(model_dict, initial={t: 0.0, s: s0, h: h0, f:f0})

    fit = Fit(ode_model, t=time, s=conv, h=None, f=None)
    fit_result = fit.execute()
    print(fit_result)


def solve_eq():
    a, b, c, t = sp.symbols("a, b, c, t", real=True, positive=True)
    x = sp.Symbol("x", real=True, positive=True)
    eq = -b*sp.log(x) + a*(1-x) -1/c*(1-sp.exp(-1*c*t))
    s = sp.solve(eq, x)
    print(s)


def func(t, a, b, c):
    s = b*lambertw(a*np.exp((a - 1/c + np.exp(-c*t)/c)/b)/b)/a
    return (s.real).astype(float)


def func_complex(t, b, c):
    s = b*lambertw(np.exp((c*np.exp(c*t) - np.exp(c*t) + 1)*np.exp(-c*t)/(b*c))/b)
    return s


def train_test_split(data, ratio):
    number_sample = data.shape[0]
    number_test = int(number_sample*ratio)
    index_sample = [index for index in range(number_sample)]
    seed = 0
    np.random.seed(seed)
    np.random.shuffle(index_sample)
    index_test = index_sample[0:number_test]
    index_train = index_sample[number_test:]
    return index_train, index_test


def reference_model_noise(data):
    true = []
    pred = []
    # plt.figure()
    # left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    # rect = [left, bottom, width, height]
    # ax = plt.axes(rect)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.set_xlim(-2.5, 32.5)
    # ax.set_xticks(np.array([0, 5, 10, 15, 20, 25, 30]))
    # ax.set_xticklabels(np.array([0, 5, 10, 15, 20, 25, 30]),
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylim(-0.1, 1.1)
    # ax.set_yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    # ax.set_yticklabels(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylabel(r'$C/C_0$', labelpad=5, fontsize=22, fontweight='500')
    # ax.set_xlabel('Time', labelpad=5, fontsize=22)
    # ax.tick_params(length=4, width=1.5, which='major')
    # colors = ["#C8557C", "#A050A0", "#4292C6", "#5F9EA0", "#E9967A", "#F5DEB3", "#663366",
    #           "#CC0000", "#336699", "#99CCFF", "#0066CC", "#336666", "#5F9EA0", "#66CDAA", "#FF9900", "#F5DEB3"]
    p_p = []
    p_t = []
    p_p_1 = []
    p_t_1 = []
    p_p_2 = []
    p_t_2 = []
    for index in range(data.shape[0]):
        t = np.array([2, 5, 9, 15, 22, 30])
        conv = (1 - data[index, 5:])
        try:
            params = curve_fit(func, t, conv)
            func_fitted = lambda t: func(t, params[0][0] + np.random.uniform(-0.1, 0.1, 1)[0],
                                         params[0][1] + np.random.uniform(-0.005, 0.005, 1)[0],
                                         params[0][2] + np.random.uniform(-0.1, 0.1, 1)[0])
            conv_pred = list(map(func_fitted, t))
            p_t.append(params[0][0])
            p_p.append(params[0][0] + np.random.uniform(-0.1, 0.1, 1)[0])
            p_t_1.append(params[0][1])
            p_p_1.append(params[0][1] + np.random.uniform(-0.005, 0.005, 1)[0])
            p_t_2.append(params[0][2])
            p_p_2.append(params[0][2] + np.random.uniform(-0.1, 0.1, 1)[0])
            Flag = True
            if sum(conv_pred) == 0:
                pos = -1
                for i in range(conv.shape[0]):
                    if conv[i] < 0.18:
                        pos = i
                        break
                if pos != -1:
                    params = curve_fit(func, t[0:pos], conv[0:pos])
                    func_fitted = lambda t: func(t, params[0][0], params[0][1], params[0][2])
                    conv_pred = list(map(func_fitted, t[0:pos]))
                    # print(params)
                    # print(test_ori[index, 1:5])
                    conv_pred = conv_pred + [conv[pos] for i in range(6 - len(conv_pred))]
                else:
                    Flag = False
                    print(conv)
            if Flag:
                true.append(conv * 100)
                pred.append(np.array(conv_pred) * 100)
        except:
            pass

    #     if index <= 8:
    #         ax.scatter([0, 2, 5, 9, 15, 22, 30], [1] + conv.tolist(), label='Measured', c=colors[index])
    #         ax.plot([0, 2, 5, 9, 15, 22, 30], [1] + conv_pred, label='Fitted', c=colors[index])
    # ax.scatter([23], [1], c='black')
    # ax.text(x=24, y=1, s='Measured', va='center', fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # ax.plot([22.5, 23.5], [0.9, 0.9], c='black')
    # ax.text(x=24, y=0.9, s='Fitted', va='center', fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # plt.show()
    return true, pred, p_p, p_t, p_p_1, p_t_1, p_p_2, p_t_2


def reference_model(data):
    true = []
    pred = []
    # plt.figure()
    # left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    # rect = [left, bottom, width, height]
    # ax = plt.axes(rect)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.set_xlim(-2.5, 32.5)
    # ax.set_xticks(np.array([0, 5, 10, 15, 20, 25, 30]))
    # ax.set_xticklabels(np.array([0, 5, 10, 15, 20, 25, 30]),
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylim(-0.1, 1.1)
    # ax.set_yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    # ax.set_yticklabels(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylabel(r'$C/C_0$', labelpad=5, fontsize=22, fontweight='500')
    # ax.set_xlabel('Time', labelpad=5, fontsize=22)
    # ax.tick_params(length=4, width=1.5, which='major')
    # colors = ["#C8557C", "#A050A0", "#4292C6", "#5F9EA0", "#E9967A", "#F5DEB3", "#663366",
    #           "#CC0000", "#336699", "#99CCFF", "#0066CC", "#336666", "#5F9EA0", "#66CDAA", "#FF9900", "#F5DEB3"]

    for index in range(data.shape[0]):
        t = np.array([2, 5, 9, 15, 22, 30])
        conv = (1 - data[index, 5:])
        try:
            params = curve_fit(func, t, conv)
            func_fitted = lambda t: func(t, params[0][0],
                                         params[0][1],
                                         params[0][2])
            conv_pred = list(map(func_fitted, t))

            Flag = True
            if sum(conv_pred) == 0:
                pos = -1
                for i in range(conv.shape[0]):
                    if conv[i] < 0.18:
                        pos = i
                        break
                if pos != -1:
                    params = curve_fit(func, t[0:pos], conv[0:pos])
                    func_fitted = lambda t: func(t, params[0][0], params[0][1], params[0][2])
                    conv_pred = list(map(func_fitted, t[0:pos]))
                    # print(params)
                    # print(test_ori[index, 1:5])
                    conv_pred = conv_pred + [conv[pos] for i in range(6 - len(conv_pred))]
                else:
                    Flag = False
                    print(conv)
            if Flag:
                true.append(conv * 100)
                pred.append(np.array(conv_pred) * 100)
        except:
            pass

    #     if index <= 8:
    #         ax.scatter([0, 2, 5, 9, 15, 22, 30], [1] + conv.tolist(), label='Measured', c=colors[index])
    #         ax.plot([0, 2, 5, 9, 15, 22, 30], [1] + conv_pred, label='Fitted', c=colors[index])
    # ax.scatter([23], [1], c='black')
    # ax.text(x=24, y=1, s='Measured', va='center', fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # ax.plot([22.5, 23.5], [0.9, 0.9], c='black')
    # ax.text(x=24, y=0.9, s='Fitted', va='center', fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # plt.show()
    return true, pred


def EP_plot_p(true, pred, type):
    mpl.rcParams["mathtext.fontset"] = 'custom'
    mpl.rcParams["mathtext.bf"] = "Arial:bold"
    mpl.rcParams["mathtext.default"] = 'regular'
    plt.figure()
    # mpl.rcParams["markers.fillstyle"] = 'none'
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    if type == 'c':
        ax.set_xlim(-15.5, 1.5)
        ax.set_ylim(-15.5, 1.5)
        ax.set_xticks(np.array([-15, -10, -5, 0]))
        ax.set_xticklabels(np.array([-15, -10, -5, 0]),
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_yticks(np.array([-15, -10, -5, 0]))
        ax.set_yticklabels(np.array([-15, -10, -5, 0]),
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_ylabel('Param c (with noise)', labelpad=5, fontsize=22, fontweight='500',
                      fontfamily='Arial')
        ax.set_xlabel('Param c', labelpad=5, fontsize=22,
                      fontfamily='Arial')
        plt.plot([-15.5, 1.5], [-15.5, 1.5], c='#C26275', linestyle='dashed', zorder=0)
        ax.text(x=-5, y=-12, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=-5, y=-13.5, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=-5, y=-15, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    elif type == 'b':
        ax.set_xlim(-50, 450)
        ax.set_ylim(-50, 450)
        ax.set_xticks(np.array([0, 100, 200, 300, 400]))
        ax.set_xticklabels(np.array([0, 100, 200, 300, 400]),
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_yticks(np.array([0, 100, 200, 300, 400]))
        ax.set_yticklabels(np.array([0, 100, 200, 300, 400]),
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_ylabel('Param b (with noise)', labelpad=5, fontsize=22, fontweight='500',
                      fontfamily='Arial')
        ax.set_xlabel('Param b', labelpad=5, fontsize=22,
                      fontfamily='Arial')
        plt.plot([-50, 450], [-50, 450], c='#C26275', linestyle='dashed', zorder=0)
        ax.text(x=275, y=75, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=275, y=25, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=275, y=-25, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    elif type == 'a':
        ax.set_xlim(-450, 50)
        ax.set_ylim(-450, 50)
        ax.set_xticks(np.array([-400, -300, -200, -100, 0]))
        ax.set_xticklabels(np.array([-400, -300, -200, -100, 0]),
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_yticks(np.array([-400, -300, -200, -100, 0]))
        ax.set_yticklabels(np.array([-400, -300, -200, -100, 0]),
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_ylabel('Param a (with noise)', labelpad=5, fontsize=22, fontweight='500',
                      fontfamily='Arial')
        ax.set_xlabel('Param a', labelpad=5, fontsize=22,
                      fontfamily='Arial')
        plt.plot([-450, 50], [-450, 50], c='#C26275', linestyle='dashed', zorder=0)
        ax.text(x=-125, y=-325, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=-125, y=-375, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=-125, y=-425, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    else:
        raise ValueError
    ax.tick_params(length=4, width=1.5, which='major')
    plt.scatter(true, pred, marker='o', s=125, lw=0.2, alpha=0.6, c='#B7E1FC', edgecolor='black', zorder=1)  # #D2D2FF
    # plt.savefig('/home/huatianwei/Code/Code for MEARML model/Figures/R5c.svg', dpi=600)
    plt.show()

    # EP_plot_1(p_t_2, p_p_2)


def EP_plot(true, pred):
    mpl.rcParams["mathtext.fontset"] = 'custom'
    mpl.rcParams["mathtext.bf"] = "Arial:bold"
    mpl.rcParams["mathtext.default"] = 'regular'
    plt.figure()
    # mpl.rcParams["markers.fillstyle"] = 'none'
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([0, 20, 40, 60, 80, 100],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([0, 20, 40, 60, 80, 100],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Predicted Conc.' + ' (%)', labelpad=5, fontsize=22, fontweight='500',
                  fontfamily='Arial')
    ax.set_xlabel('Measured Conc.' + ' (%)', labelpad=5, fontsize=22,
                  fontfamily='Arial')
    ax.tick_params(length=4, width=1.5, which='major')
    plt.scatter(true, pred, marker='o', s=125, lw=0.2, alpha=0.6, c='#B7E1FC', edgecolor='black', zorder=1)  # #D2D2FF
    plt.plot([-10, 110], [-10, 110], c='#C26275', linestyle='dashed', zorder=0)
    ax.text(x=65, y=15, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
            fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=65, y=5, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
            fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=65, y=-5, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
            fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    plt.savefig('/home/huatianwei/Code/Code for MEARML model/Figures/R5e.svg', dpi=600)
    plt.show()


if __name__ == '__main__':
    '''reference model'''
    data_ori = np.array(pd.read_csv('Fenton-data.csv'))
    data = np.array(pd.read_csv('Fenton-data.csv'))[:, 1:]
    mask_train, mask_test = train_test_split(data, 0.2)
    data = data[mask_test, :]
    test_ori = data_ori[mask_test, :]

    true, pred = reference_model(data)
    true = np.array(true).reshape(1, -1)[0]
    pred = np.array(pred).reshape(1, -1)[0]
    true_filter = []
    pred_filter = []
    for index in range(pred.shape[0]):
        if np.isnan(pred[index]):
            pass
        else:
            true_filter.append(true[index])
            pred_filter.append(pred[index])
    EP_plot(true_filter, pred_filter)

    true, pred, p_p, p_t, p_p_1, p_t_1, p_p_2, p_t_2 = reference_model_noise(data)
    EP_plot_p(p_t, p_p, 'a')
    EP_plot_p(p_t_1, p_p_1, 'b')
    EP_plot_p(p_t_2, p_p_2, 'c')

    true= np.array(true).reshape(1, -1)[0]
    pred = np.array(pred).reshape(1, -1)[0]
    true_filter = []
    pred_filter = []
    for index in range(pred.shape[0]):
        if np.isnan(pred[index]):
            pass
        else:
            true_filter.append(true[index])
            pred_filter.append(pred[index])
    EP_plot(true_filter, pred_filter)
    '''reference model'''
















