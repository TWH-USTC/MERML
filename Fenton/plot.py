import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib as mpl
from scipy.stats import pearsonr
from itertools import product
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import ColorbarBase
import palettable
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# mpl.rcParams["markers.fillstyle"] = 'none'
mpl.rcParams["mathtext.fontset"] = 'custom'
mpl.rcParams["mathtext.bf"] = "Arial:bold"
mpl.rcParams["mathtext.default"] = 'regular'
color = ["#C8557C", "#A050A0", "#4292C6", "#5F9EA0", "#E9967A", "#F5DEB3", "#663366",
             "#CC3366", "#DB7093", "#CC0000",
             "#336699", "#99CCFF", "#0066CC", "#336666", "#5F9EA0", "#66CDAA", "#FF9900", "#F5DEB3"]


def range_brace(x_min, x_max, mid=0.5, beta1=75, beta2=75, height=-0.05,
                initial_divisions=11, resolution_factor=1):
    x0 = np.array(())
    tmpx = np.linspace(0, 0.5, initial_divisions)
    tmp = beta1**2 * (np.exp(beta1*tmpx)) * (1-np.exp(beta1*tmpx)) / np.power((1+np.exp(beta1*tmpx)),3)
    tmp += beta2**2 * (np.exp(beta2*(tmpx-0.5))) * (1-np.exp(beta2*(tmpx-0.5))) / np.power((1+np.exp(beta2*(tmpx-0.5))),3)
    for i in range(0, len(tmpx)-1):
        t = int(np.ceil(resolution_factor*max(np.abs(tmp[i:i+2]))/float(initial_divisions)))
        x0 = np.append(x0, np.linspace(tmpx[i],tmpx[i+1],t))
    x0 = np.sort(np.unique(x0)) # sort and remove dups
    # half brace using sum of two logistic functions
    y0 = mid*2*((1/(1.+np.exp(-1*beta1*x0)))-0.5)
    y0 += (1-mid)*2*(1/(1.+np.exp(-1*beta2*(x0-0.5))))
    # concat and scale x
    x = np.concatenate((x0, 1-x0[::-1])) * float((x_max-x_min)) + x_min
    y = np.concatenate((y0, y0[::-1])) * float(height)
    return (x, y)


def corr_plot(x_cof, name, types='fill', front_raito=1, label_axis="off"):
    colorlist = palettable.colorbrewer.sequential.Blues_9.hex_colors[0:2]
    colorlist = ['#E4F4FE', '#CFEBFD', '#B2E0FC', '#87cefa', '#7CCAFA'] #, '#61BFF9'
    # colorlist.reverse()
    print(colorlist)

    colormap = mpl.colors.LinearSegmentedColormap.from_list('323', colorlist, N=256, gamma=1.0)
    norm = mpl.colors.Normalize(vmax=x_cof.max(), vmin=x_cof.min())
    value2color = lambda x: colormap(norm(x))
    colors = np.array([value2color(cor) for cor in x_cof.reshape(1, -1).tolist()])\
        .reshape(x_cof.shape[0], x_cof.shape[1], 4)
    fill_colors = colors

    x_cof = np.round(x_cof, 2)

    size = x_cof
    n = size.shape[0]

    fig = plt.figure()
    gs = gridspec.GridSpec(n, n)
    gs.update(wspace=0.0, hspace=0.0, left=0.11, bottom=0.18, right=0.8, top=0.96)

    for i, j in product(range(0, n, 1), range(0, n, 1)):
        if types == "fill":
            ax = plt.subplot(gs[i, j])
            ax.set_facecolor(fill_colors[i, j])
            [ax.spines[_].set_color('w') for _ in ['right', 'top', 'left', 'bottom']]

            ax.text(0.5, 0.5, x_cof[i, j],
                        fontdict={'family': 'Arial', 'weight':'500', 'color':'#000000'},  # args
                        fontsize=14,  # c_arg
                        horizontalalignment='center', verticalalignment='center_baseline')
            if j == 0:
                ax.text(-0.1, 0.5, s=name[i], horizontalalignment='right', va='center',
                        fontdict={'family': 'Arial', 'size': '18', 'weight': '500'}, rotation=30)
            if i == 5:
                ax.text(0.5, -0.1, s=name[j], horizontalalignment='center', va='top',
                        fontdict={'family': 'Arial', 'size': '18', 'weight': '500'}, rotation=30)

            ax.set_xticks([])
            ax.set_yticks([])

    cbar_ax = fig.add_axes([0.82, 0.18, 0.03, 0.78])
    ColorbarBase(cbar_ax, cmap=colormap, norm=norm)
    ColorbarBase(cbar_ax, cmap=colormap, norm=norm)
    cf = ColorbarBase(cbar_ax, cmap=colormap, norm=norm)
    ax_cf = cf.ax
    ax_cf.tick_params(length=4, width=1.5, which='major')
    # ax_cf.set_ylim(0.46, 1.1)
    ax_cf.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_cf.set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'],
                       fontfamily='Arial', fontsize=16, fontweight='300')
    plt.text(x=3.6, y=0.8, s='Pearson codfficient',
             rotation='vertical', fontfamily='Arial',
             fontsize=22, fontweight='500', va='center')
    cf.outline.set_linewidth(1.5)
    plt.savefig('Figures/2b.svg', dpi=600)
    plt.show()


def calculate_kinetic(pca, mm_scaler):
    data = [[], [], [], []]
    rate = []
    zero_func = lambda k, t: 1-k*t
    first_func = lambda k, t: np.exp(-k*t)
    second_func = lambda k, t: 1/(1+k*t)
    third_func  = lambda k,t: np.sqrt(1/(1+k*t))
    kinetic_func = [zero_func, first_func, second_func, third_func]
    # mixed order
    # s = 0.5
    # c = (1 + s) / (np.exp(k * t) + s)

    kinetic_parameters = [np.arange(0, 1, 0.001), np.arange(0, 10, 0.0001),
                          np.arange(0, 100, 0.0001), np.arange(0, 1000, 0.001)]
    TIME = [0, 2, 5, 9, 15, 22, 30]
    for index in range(len(kinetic_func)):
        for k in kinetic_parameters[index]:
            data_temp = []
            for t in TIME:
                c = kinetic_func[index](k, t)
                if c >= 0:
                    pass
                else:
                    c = 0
                data_temp.append(c)
            data[index].append(data_temp)
        data[index] = 1 - np.array(data[index]).reshape(-1, 7)
        rate_0 = (data[index][:, :-1] - data[index][:, 1:]) / [2, 3, 4, 6, 7, 8]
        rate_0 = pca.transform(mm_scaler.transform(rate_0))
        rate.append(rate_0)
    np.save('Result/rate.npy', rate)


def dynamic(data):
    global color
    curves = (1 - data[:, -7:])
    rate = (curves[:, :-1] - curves[:, 1:]) / [2, 3, 4, 6, 7, 8]
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(rate)
    rate = mm_scaler.transform(rate)
    pca = PCA(n_components=2)
    pca.fit(rate)
    print(pca.explained_variance_ratio_)
    rate_rd = pca.transform(rate)

    # point1 = pca.transform(mm_scaler.transform(np.array([1, 0, 0, 0, 0, 0]).reshape(1, -1)/ [2, 3, 4, 6, 7, 8]))
    # point2 = pca.transform(mm_scaler.transform(np.array([0, 0, 0, 0, 0, 0]).reshape(1, -1)/ [2, 3, 4, 6, 7, 8]))

    calculate_kinetic(pca, mm_scaler) # generate the rate.npy
    rate = np.load('Result/rate.npy', allow_pickle=True)

    plt.figure()
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.set_xlim(-0.8, 2)
    ax.set_ylim(-1, 1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_yticks([-0.8, -0.4, 0.0, 0.4, 0.8, 1.2])
    ax.set_yticklabels(['-0.8', '-0.4', '0.0', '0.4', '0.8', '1.2'],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    ax.set_xticklabels(['-0.5', '0.0', '0.5', '1.0', '1.5'],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('PCA 2', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('PCA 1', labelpad=5, fontsize=22, fontweight='500')
    ax.tick_params(length=4, width=1.5, which='major')
    plt.scatter(rate_rd[:, 0], rate_rd[:, 1], c='#87cefa', zorder=-1, edgecolors='black',
                alpha=0.6, linewidths=0.1, s=100)
    plt.plot(rate[0][:, 0], rate[0][:, 1], c='#A9D18E', zorder=1, lw=1.5, ls='--')
    plt.plot(rate[1][:, 0], rate[1][:, 1], c='#B39BC7', zorder=0, lw=1.5, ls='--')
    plt.plot(rate[2][:, 0], rate[2][:, 1], c='#DF7189', zorder=0, lw=1.5, ls='--')
    plt.plot(rate[3][:, 0], rate[3][:, 1], c='#F4B183', zorder=0, lw=1.5, ls='--')
    # ax.axhline(0.0, linestyle='--', c='#4D4D4D', lw=1.5)
    # ax.axvline(0.0, linestyle='--', c='#4D4D4D', lw=1.5)
    plt.scatter(0.675, 0.77, c='#87cefa', zorder=-1, edgecolors='black',
                alpha=0.6, linewidths=0.25, s=100)
    ax.text(x=0.75, y=0.75, s='Fenton reaction',
            fontdict={'family': 'Arial', 'size': '16', 'weight': '500'}, va='center')
    plt.plot(np.array([0.8, 0.85])-0.15, np.array([1.0, 1.0])*1.35, c='#A9D18E', zorder=0, lw=1.5, ls='-')
    plt.plot(np.array([0.8, 0.85])-0.15, np.array([1.0, 1.0])*1.2, c='#B39BC7', zorder=0, lw=1.5, ls='-')
    plt.plot(np.array([0.8, 0.85])-0.15, np.array([1.0, 1.0])*1.05, c='#DF7189', zorder=0, lw=1.5, ls='-')
    plt.plot(np.array([0.8, 0.85])-0.15, np.array([1.0, 1.0])*0.9, c='#F4B183', zorder=0, lw=1.5, ls='-')

    ax.text(x=0.75, y=1.35, s='zeroth order reaction', fontdict={'family': 'Arial', 'size': '16', 'weight': '500'}, va='center')
    ax.text(x=0.75, y=1.2, s='first order reaction', fontdict={'family': 'Arial', 'size': '16', 'weight': '500'}, va='center')
    ax.text(x=0.75, y=1.05, s='second order reaction', fontdict={'family': 'Arial', 'size': '16', 'weight': '500'}, va='center')
    ax.text(x=0.75, y=0.9, s='third order reaction', fontdict={'family': 'Arial', 'size': '16', 'weight': '500'}, va='center')

    ax.plot(range_brace(0.9, 1.35)[1] + 0.6, range_brace(0.9, 1.35)[0], color='black', lw=1.5, clip_on=False)

    ax.text(-0.65, 1.135 + 0.04, s='-', fontdict={'family': 'Arial', 'size': '16', 'weight': '500'},
            va='center_baseline')
    ax.plot([0.065-0.65, 0.185-0.65], [1.1175+0.04, 1.1175+0.04], c='black', zorder=0, lw=1.5)
    ax.text(0.06-0.65, 1.2+0.04, s='dx', fontdict={'family': 'Arial', 'size': '16', 'weight': '500'},
            va='center_baseline')
    ax.text(0.07-0.65, 1.05+0.04, s='dt', fontdict={'family': 'Arial', 'size': '16', 'weight': '500'},
            va='center_baseline')
    ax.text(x=0.195-0.65, y=1.14+0.04, s=' = ' + r'$k[x]^n$' +', ' + 'k '+ r'$∈$'+' [0,' + U'$\infty$' + ')',
            fontdict={'family': 'Arial', 'size': '16', 'weight': '500'},
            va='center_baseline')

    ax.text(-0.1, -0.75, s='k ' + r'$→$',
            fontdict={'family': 'Arial', 'size': '16', 'weight': '500'}, va='center')
    ax.text(0.15, -0.76, s=U'$\infty$',
            fontdict={'family': 'Arial', 'size': '20', 'weight': '500'}, va='center')
    ax.text(-0.6, -0.75, s='k ' + r'$→$' + ' 0',
            fontdict={'family': 'Arial', 'size': '16', 'weight': '500'}, va='center')

    ax.annotate("", xytext=(-0.45, -0.7), xy=(-0.57, -0.50), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xytext=(0.05, -0.7), xy=(-0.1, -0.32), arrowprops=dict(arrowstyle="->"))
    plt.savefig('Figures/2a.svg',dpi=600)
    plt.show()


def autoregression(data):
    global color
    curve = data[:, -6:]
    r_mat = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            conv_prvious = curve[:, i]
            conv_current = curve[:, j]
            r_mat[i, j] = pearsonr(conv_prvious, conv_current)[0]
    corr_plot(r_mat, name=[r'$C_2$', r'$C_5$', r'$C_9$', r'$C_{15}$', r'$C_{22}$', r'$C_{30}$'], types='fill')


def split_plots():  # Sketch Map of kinetic profile example in training and test or validation sets
    plt.figure(dpi=300)
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(-2.5, 32.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.set_yticklabels(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Conc.', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('Time (min)', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')

    colors = ['#9DC3E6', '#A9D18E', '#F4B183', '#B39BC7']

    x = np.arange(0, 31, 1)
    y = 1 / (0.045 * x + 1)
    t = [0, 2, 5, 9, 15, 22, 30]
    c = 1 / (np.array(t) * 0.045 + 1)
    ax.scatter(t, c, zorder=2, c=colors[0], s=50)
    ax.plot(x, y, zorder=1, color=colors[0], lw=1.5)

    x = np.arange(0, 31, 1)
    y = 1 / (0.08 * x + 1)
    t = [0, 2, 5, 9, 15, 22, 30]
    c = 1 / (np.array(t) * 0.08 + 1)
    std = y * 0.075
    ax.scatter(t, c, zorder=2, c=colors[1], s=50)
    ax.plot(x, y, zorder=1, color=colors[1], lw=1.5)
    ax.fill_between(x, y - std, y + std, alpha=0.4, zorder=0, color=colors[1])

    x = np.arange(0, 31, 1)
    y = 1 / (0.15 * x + 1)
    t = [0, 5, 15, 30]
    c = 1 / (np.array(t) * 0.15 + 1)
    ax.scatter(t, c, zorder=2, c=colors[2], s=50)
    ax.plot(x, y, zorder=1, color=colors[2], lw=1.5)

    x = np.arange(0, 31, 1)
    y = 1 / (0.4 * x + 1)
    t = [0, 2, 5, 9, 15, 22, 30]
    c = 1 / (np.array(t) * 0.4 + 1)
    ax.scatter(t, c, zorder=2, c=colors[3], s=50, marker='D')
    ax.plot(x, y, zorder=1, color=colors[3], lw=1.5)

    x = 13
    ax.plot([x, x+3], [1.0, 1.0], lw=1.5, color=colors[0])
    ax.scatter(x+1.5, 1.0, s=50, c=colors[0])
    ax.text(x+3.5, 1.0, s='original', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'},
            ha='left', va='center')

    ax.plot([x, x+3], [0.9, 0.9], lw=1.5, color=colors[1])
    ax.scatter(x+1.5, 0.9, s=50, c=colors[1])
    ax.text(x+3.5, 0.9, s='with Gaussian error', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'},
            ha='left', va='center')

    ax.plot([x, x+3], [0.8, 0.8], lw=1.5, color=colors[2])
    ax.scatter(x+1.5, 0.8, s=50, c=colors[2])
    ax.text(x+3.5, 0.8, s='with less points', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'},
            ha='left', va='center')

    ax.plot([x, x+3], [0.7, 0.7], lw=1.5, color=colors[3])
    ax.scatter(x+1.5, 0.7, s=50, c=colors[3], marker='D')
    ax.text(x+3.5, 0.7, s='Test set', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'},
            ha='left', va='center')

    ax.plot(range_brace(0, 6)[1] * 30 + x-1, range_brace(0, 6)[0] / 30 + 0.8, color='black', lw=1.0, clip_on=False)
    ax.text(x-9, 0.95, s='training', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'},
            ha='left', va='center')
    ax.text(x-7.5, 0.85, s='set', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'},
            ha='left', va='center')
    # ax.spines['right'].set_visible(False)  # 关闭子图2中底部脊
    # ax.spines['top'].set_visible(False)  ##关闭子图1中顶部脊
    plt.savefig('Figures/4a.svg', dpi=600)
    plt.show()


def plot_model_compare():
    data = np.load('Result/metrics_model_compare.npy', allow_pickle=True)
    print(data)
    mae = data[0, :]*100
    std = data[1, :]*100

    plt.figure()
    fig = plt.figure()

    c = ['#9DC3E6', '#A9D18E', '#F4B183']
    ax3 = fig.add_axes([0.18, 0.18, 0.78, 0.78], zorder=0)
    ax3.set_xlim(0.3, 5.7)
    ax3.set_ylim(0, 1)
    rect = mpl.patches.Rectangle(xy=(0.5, 0), width=1, height=1, facecolor=c[0], edgecolor='black',
                                 linewidth=0, alpha=0.15, zorder=2)
    ax3.add_patch(rect)
    ax3.text(1, 0.85, 'Our'+'\n'+'model', zorder=3, fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'},
            ha='center', va='center')
    rect = mpl.patches.Rectangle(xy=(1.5, 0), width=2, height=1, facecolor=c[1], edgecolor='black',
                                 linewidth=0, alpha=0.15, zorder=2)
    ax3.add_patch(rect)
    ax3.text(2.5, 0.85, 'Ablated' + '\n' + 'model', zorder=3,
             fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'},
             ha='center', va='center')
    rect = mpl.patches.Rectangle(xy=(3.5, 0), width=2, height=1, facecolor=c[2], edgecolor='black',
                                 linewidth=0, alpha=0.15, zorder=2)
    ax3.add_patch(rect)
    ax3.text(4.5, 0.85, 'Commonly used' + '\n' + 'model', zorder=3,
             fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'},
             ha='center', va='center')
    ax3.spines['bottom'].set_visible(False)  # 关闭子图2中底部脊
    ax3.spines['top'].set_visible(False)  ##关闭子图1中顶部脊
    ax3.spines['left'].set_visible(False)  # 关闭子图2中底部脊
    ax3.spines['right'].set_visible(False)  ##关闭子图1中顶部脊
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax2 = fig.add_axes([0.18, 0.55, 0.78, 0.41], zorder=1, facecolor='none')
    ax1 = fig.add_axes([0.18, 0.18, 0.78, 0.35], zorder=1, facecolor='none')
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['top'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax1.set_xlim(0.3, 5.7)
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_xticklabels(['ERML', 'RML', 'ML', 'PDP', 'MB'],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax2.set_xlim(0.3, 5.7)
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xticklabels(['ERML', 'RML', 'ML', 'PDP', 'MB'],
                        fontfamily='Arial', fontsize=18, fontweight='500')

    ax1.set_ylim(3, 5.5)
    ax1.set_yticks([3, 4, 5])
    ax1.set_yticklabels([3, 4, 5],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax2.set_ylim(7.5, 16)
    ax2.set_yticks([8, 12, 16])
    ax2.set_yticklabels([8, 12, 16],
                        fontfamily='Arial', fontsize=18, fontweight='500')
    ax1.set_xlabel('Model', labelpad=12, fontsize=22)
    ax1.tick_params(length=4, width=1.5, which='major')
    ax2.tick_params(length=4, width=1.5, which='major')

    error_attri = dict(elinewidth=1, ecolor="black", capsize=5)
    ax1.bar([1, 2, 3, 4, 5], mae, color="#87CEFA", width=0.6, align="center",
             yerr=std, error_kw=error_attri, linewidth=1, edgecolor='black', alpha=0.6)
    ax2.bar([1, 2, 3, 4, 5], mae, color="#87CEFA", width=0.6, align="center",
            yerr=std, error_kw=error_attri, linewidth=1, edgecolor='black', alpha=0.6)

    d = .85  # 设置倾斜度
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12.5,
                  linestyle='none', color='k', mec='k', mew=1.5, clip_on=False)
    ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
    ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
    ax2.spines['bottom'].set_visible(False)  # 关闭子图2中底部脊
    ax1.spines['top'].set_visible(False)  ##关闭子图1中顶部脊
    ax2.set_xticks([])
    ax1.text(x=-0.4, y=5.5, s='MAE of Conc. (%)', rotation= 90, ha='center', va='center',
             fontdict={'fontfamily': 'Arial', 'fontsize': '22', 'fontweight': '500'})
    plt.savefig('Figures/4b.svg', dpi=600)
    plt.show()


def plot_model_noise():
    data = np.load('Result/metrics_noise.npy')[:, :, :, 0]
    print(data.shape)
    num = 7
    data_1 = data[0, 0:num, :]*100
    data_2 = data[1, 0:num, :]*100
    data_3 = data[2, 0:num, :]*100
    data_list = [data_1, data_2, data_3]
    labels = ['ERML', 'RML', 'ML']

    plt.figure()
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(-0.5, 6.5)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylim(4, 8)
    ax.set_yticks(np.array([4, 5,  6, 7, 8]))
    ax.set_yticklabels(np.array([4, 5,  6, 7, 8]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('MAE of Conc.' + ' (%)', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('SD of Gaussian error (%)', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')

    c = ['#9DC3E6', '#A9D18E', '#F4B183']
    # marker = ['o', '^', 'D']
    for index in range(0, 3, 1):
        index = 2 - index
        plt.plot(np.arange(0, num, 1), np.mean(data_list[index], axis=1), c=c[index], marker='o',
                 label=labels[index], markersize=6, linewidth=1.5) #
    # plt.hlines(y=5, xmin=-0.5, xmax=10.5, linewidth=1.5, linestyles='--', color='#767171',)
    plt.legend(frameon=False, prop={'family': 'Arial', 'size': '18', 'weight': '500'})
    plt.savefig('Figures/4c.svg', dpi=600)
    plt.show()


def plot_model_points():
    data = np.load('Result/metrics_points.npy', allow_pickle=True)
    x = []
    for index in range(0, 5, 1):
        mean = np.mean(np.array(data[index]), axis=2)[:, :, 0]*100
        x.append(mean[:, 0])
        x.append(mean[:, 1])
        x.append(mean[:, 2])
    plt.figure()
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.set_ylim(0, 35)
    ax.set_yticks(np.array([0, 5, 10, 15, 20, 25, 30, 35]))
    ax.set_yticklabels(np.array([0, 5, 10, 15, 20, 25, 30, 35]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('MAE of Conc.' + ' (%)', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('Concentration-time points', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')

    c = ['#BBE6FF', '#D8EDD9', '#FCEDE2']*5

    ax.hlines(y=0.05, xmin=0, xmax=20)
    pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    # marker = ['o', '^', 'D']
    # hatch = ['///', '', '---']
    for index in range(len(x)):
        bprops = {'facecolor':c[index], 'color':'black', 'lw':1.0, } # 'hatch':hatch[index%3]
        ax.boxplot(x[index], positions=[pos[index]], widths=0.6, patch_artist=True, boxprops=bprops
                    , flierprops={'markersize':6, 'lw':1.0, }, #'marker': marker[index%3]
                    medianprops={'linestyle':'-','linewidth':1.0,'color':'#DC143C'})

    ax.set_xlim(0, 20)
    ax.set_xticks([2, 6, 10, 14, 18])
    ax.set_xticklabels(['1', '2', '3', '4', '5'],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    # plt.hlines(y=5, xmin=0, xmax=20, linewidth=1.5, linestyles='--', color='#767171', )
    # mpl.rcParams["markers.fillstyle"] = 'none'
    # plt.scatter(13.25, 31.75, marker=marker[0], s=60, lw=1.0, c='black')
    # plt.scatter(13.25, 29.25, marker=marker[1], s=60, lw=1.0, c='black')
    # plt.scatter(13.25, 26.75, marker=marker[2], s=60, lw=1.0, c='black')
    rect = mpl.patches.Rectangle(xy=(14, 31), width=1, height=1.5, facecolor=c[0], edgecolor='black',
                                 linewidth=1.5)
    ax.add_patch(rect)
    rect_1 = mpl.patches.Rectangle(xy=(14, 28.5), width=1, height=1.5, facecolor=c[1], edgecolor='black',
                                   linewidth=1.5)
    ax.add_patch(rect_1)
    rect_2 = mpl.patches.Rectangle(xy=(14, 26), width=1, height=1.5, facecolor=c[2], edgecolor='black',
                                   linewidth=1.5)
    # ax.plot([14.1, 14.4], [31, 32.5], c='black')
    # ax.plot([14.4, 14.7], [31, 32.5], c='black')
    # ax.plot([14.7, 15.0], [31, 32.5], c='black')
    # ax.plot([14, 15], [26.5, 26.5], c='black')
    # ax.plot([14, 15], [27, 27], c='black')
    # ax.plot([14, 15], [27.5, 27.5], c='black')
    ax.add_patch(rect_2)
    plt.text(x=15.5, y=31, s='ERML', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'})
    plt.text(x=15.5, y=28.5, s='RML', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'})
    plt.text(x=15.5, y=26, s='ML', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'})
    # plt.scatter(15, 23.25, marker='o', s=50, lw=1.0, facecolors='None', edgecolor='black')
    # plt.text(x=16, y=22.5, s='Flier', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'})
    # plt.plot([14.5, 15.5], [20.75, 20.75], linestyle='-', lw=1.0, c='#DC143C')
    # plt.text(x=16, y=20, s='Median', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'})
    plt.savefig('Figures/4d.svg', dpi=600)
    plt.show()


def lr_plot(SI=False):
    if SI:
        data_1 = np.load('Result/lrs_mcb_noise_all-5.npy', allow_pickle=True)
        data_2 = np.load('Result/lrs_mcb_noise_part-5.npy', allow_pickle=True)
        data_3 = np.load('Result/lrs_mc_noise_all-5.npy', allow_pickle=True)
        data_4 = np.load('Result/lrs_mc_noise_part-5.npy', allow_pickle=True)
        data_5 = np.load('Result/lrs_mt_noise_all-5.npy', allow_pickle=True)
        data_6 = np.load('Result/lrs_mt_noise_part-5.npy', allow_pickle=True)
    else:
        data_1 = np.load('Result/lrs_mcb_all-5.npy', allow_pickle=True)
        data_2 = np.load('Result/lrs_mcb_part-5.npy', allow_pickle=True)
        data_3 = np.load('Result/lrs_mc_all-5.npy', allow_pickle=True)
        data_4 = np.load('Result/lrs_mc_part-5.npy', allow_pickle=True)
        data_5 = np.load('Result/lrs_mt_all-5.npy', allow_pickle=True)
        data_6 = np.load('Result/lrs_mt_part-5.npy', allow_pickle=True)

    print(data_4)
    c = ['#9DC3E6', '#A9D18E', '#F4B183']
    # mpl.rcParams["markers.fillstyle"] = 'none'
    plt.figure()
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(0, 240)
    ax.set_xticks(np.array([20,  60, 100, 140, 180, 220]))
    ax.set_xticklabels(np.array([20,  60, 100, 140, 180, 220]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylim(0, 30)
    ax.set_yticks(np.array([0, 5, 10, 15, 20, 25, 30]))
    ax.set_yticklabels(np.array([0, 5, 10, 15, 20, 25, 30]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('MAE of Conc.' + ' (%)', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('Training set size', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')

    data_list = [data_1, data_3, data_5]
    labels = ['ERML', 'RML', 'ML']
    # marker = ['o', '^', 'D']
    for index in range(len(data_list)):
        data = data_list[index]
        if index == 1:
            line_color = c[1]
            rect = mpl.patches.Rectangle(xy=(60, 0), width=5, height=30, facecolor=line_color, edgecolor='black',
                                         linewidth=0, alpha=0.15, zorder=-1)
            ax.add_patch(rect)
        elif index == 0:
            line_color = c[0]
            rect = mpl.patches.Rectangle(xy=(35, 0), width=10, height=30, facecolor=line_color, edgecolor='black',
                                         linewidth=0, alpha=0.15, zorder=-1)
            ax.add_patch(rect)
        elif index == 2:
            line_color = c[2]
            rect = mpl.patches.Rectangle(xy=(55, 0), width=5, height=30, facecolor=line_color, edgecolor='black',
                                         linewidth=0, alpha=0.15, zorder=-1)
            ax.add_patch(rect)
        else:
            pass

        train = data[1, :, :]*100
        train_mean = train[:, 0]
        train_low = train[:, 1]
        train_up = train[:, 2]
        test = data[0, :, :]*100
        test_mean = test[:, 0]
        test_low = test[:, 1]
        test_up = test[:, 2]

        s = 0
        x = np.arange(s, 11, 1)*20 + 20
        # mpl.rcParams["markers.fillstyle"] = 'full'
        plt.plot(x, train_mean[s:], c=line_color, marker='o', markersize=6, linewidth=1.5, alpha=0.8,
                 label='Training - '+ labels[index])
        # mpl.rcParams["markers.fillstyle"] = 'none'
        plt.plot(x, test_mean[s:], c=line_color, marker='D', markersize=6, linewidth=1.5, alpha=0.8,
                 label='Test - '+ labels[index])
    plt.legend(frameon=False, markerscale=1.0, prop={'family': 'Arial', 'weight': 500, 'size': 16}, handlelength=1.5,
               handletextpad=0.25, loc=(0.515, 0.45))
    if SI:
        plt.savefig('Figures/s2a.svg', dpi=600)
    else:
        plt.savefig('Figures/4e.svg', dpi=600)
    plt.show()

    plt.figure(dpi=300)
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(0, 240)
    ax.set_xticks(np.array([20, 60, 100, 140, 180, 220]))
    ax.set_xticklabels(np.array([20, 60, 100, 140, 180, 220]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylim(0, 30)
    ax.set_yticks(np.array([0, 5, 10, 15, 20, 25, 30]))
    ax.set_yticklabels(np.array([0, 5, 10, 15, 20, 25, 30]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('MAE of Conc.' + ' (%)', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('Training set size', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')

    data_list = [data_2, data_4, data_6]
    labels = ['ERML', 'RML', 'ML']
    for index in range(len(data_list)):
        data = data_list[index]
        if index == 1:
            line_color = c[1]
            if SI:
                rect = mpl.patches.Rectangle(xy=(135, 0), width=10, height=30, facecolor=line_color, edgecolor='black',
                                             linewidth=0, alpha=0.15, zorder=-1)
                ax.add_patch(rect)
            else:
                rect = mpl.patches.Rectangle(xy=(135, 0), width=10, height=30, facecolor=line_color, edgecolor='black',
                                             linewidth=0, alpha=0.15, zorder=-1)
                ax.add_patch(rect)
        elif index == 0:
            line_color = c[0]
            rect = mpl.patches.Rectangle(xy=(95, 0), width=10, height=30, facecolor=line_color, edgecolor='black',
                                         linewidth=0, alpha=0.15, zorder=-1)
            ax.add_patch(rect)
        elif index == 2:
            line_color = c[2]
            rect = mpl.patches.Rectangle(xy=(115, 0), width=10, height=30, facecolor=line_color, edgecolor='black',
                                         linewidth=0, alpha=0.15, zorder=-1)
            ax.add_patch(rect)
        else:
            pass

        train = data[1, :, :] * 100
        train_mean = train[:, 0]
        train_low = train[:, 1]
        train_up = train[:, 2]
        test = data[0, :, :] * 100
        test_mean = test[:, 0]
        test_low = test[:, 1]
        test_up = test[:, 2]

        s = 0
        x = np.arange(s, 11, 1) * 20 + 20
        plt.plot(x, train_mean[s:], c=line_color, marker='o', markersize=6, linewidth=1.5, alpha=0.8,
                 label='Training - ' + labels[index])
        plt.plot(x, test_mean[s:], c=line_color, marker='D', markersize=6, linewidth=1.5, alpha=0.8,
                 label='Test - ' + labels[index])

    plt.legend(frameon=False, markerscale=1.0, prop={'family': 'Arial', 'weight': 500, 'size': 16}, handlelength=1.5,
               handletextpad=0.25, loc=(0.515, 0.45))
    if SI:
        plt.savefig('Figures/s2b.svg', dpi=600)
    else:
        plt.savefig('Figures/4f.svg', dpi=600)
    plt.show()


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
    plt.scatter(true, pred, marker='o', s=125, lw=0.2, alpha=0.6, c='#B7E1FC', edgecolor='black', zorder=1)  #  #D2D2FF
    plt.plot([-10, 110], [-10, 110], c='#C26275', linestyle='dashed', zorder=0)
    ax.text(x=65, y=15, s=r'$R^2$'+' = ' + str(np.round(r2_score(true, pred), 3)), fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=65, y=5, s='MAE = '+ str(np.round(mean_absolute_error(true, pred), 3)), fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=65, y=-5, s='RMSE = '+ str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)), fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # plt.savefig('Figures/R6b.svg', dpi=600)
    plt.show()

    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 0.12)
    from scipy.stats import norm
    sns.distplot(true - pred, bins=20, hist=True,
                 kde=True, ax=ax,
                 kde_kws={'linestyle': '-', 'linewidth': '0', 'color': "#4292C6"}, #'#A050A0' '#D2D2FF'
                 hist_kws={'width': 2, 'align': 'mid', 'color': '#BBE6FF', "edgecolor": '#000000', 'linewidth': '1.0'},
                 fit=None)
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    ax.set_xticklabels([-30, -20, -10, 0, 10, 20, 30],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_yticks([0, 0.04, 0.08, 0.12])
    ax.set_yticklabels(['0.00', '0.04', '0.08', '0.12'],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Density', labelpad=10, fontsize=22, fontweight='500')
    ax.set_xlabel('Error of Conc.' + ' (%)', labelpad=10, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')
    # plt.savefig('Figures/R2a-2.svg', dpi=600)
    plt.show()


def match_curves():
    data = np.load('Result/result_test.npy', allow_pickle=True)
    EP_plot(data[0]*100, data[1]*100)
    y_true, y_pred = data[0].reshape(-1, 6), data[1].reshape(-1, 6)
    maes = []
    for index in range(y_true.shape[0]):
        curve_true = y_true[index, :]
        curve_pred = y_pred[index, :]
        mae = mean_absolute_error(curve_true, curve_pred)
        maes.append(mae)
    mask = np.argsort(np.array(maes))
    print(mask)
    maes_sorted = np.array(maes)[mask]
    print(maes_sorted)
    plot_index = [mask[0], mask[int(len(mask)/2)+1], mask[-3]]
    # plot_index = [mask[10], mask[int(len(mask) / 2) + 2], mask[-5]]
    c = ['#9DC3E6', '#A9D18E', '#F4B183']

    plt.figure()
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(-2.5, 32.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.set_yticklabels(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Conc.', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('Time (min)', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')

    for index in range(len(plot_index)):
        cruve_true = [1] + y_true[plot_index[index], :].tolist()
        curve_pred = [1] + y_pred[plot_index[index], :].tolist()
        plt.plot([0, 2, 5, 9, 15, 22, 30], cruve_true, marker='o', c=c[index], label='Measured', markersize=8)
        plt.plot([0, 2, 5, 9, 15, 22, 30], curve_pred, marker='D', c=c[index], label='Predicted', markersize=8)
    plt.legend(frameon=False, prop={'family': 'Arial', 'size': '14', 'weight': '500'})

    ax.plot(range_brace(0, 3)[1]*30+20.5, range_brace(0, 3)[0]/30+0.925, color='black', lw=1.0, clip_on=False)
    ax.plot(range_brace(0, 3)[1]*30+20.5, range_brace(0, 3)[0]/30+0.75, color='black', lw=1.0, clip_on=False)
    ax.plot(range_brace(0, 3)[1]*30+20.5, range_brace(0, 3)[0]/30+0.57, color='black', lw=1.0, clip_on=False)
    ax.text(18.5, 0.975, s='low error', fontdict={'fontfamily': 'Arial', 'fontsize': '14', 'fontweight': '500'},
            ha='right', va='center')
    ax.text(18.5, 0.8, s='medium error', fontdict={'fontfamily': 'Arial', 'fontsize': '14', 'fontweight': '500'},
            ha='right', va='center')
    ax.text(18.5, 0.62, s='high error', fontdict={'fontfamily': 'Arial', 'fontsize': '14', 'fontweight': '500'},
            ha='right', va='center')
    plt.savefig('Figures/5b.svg', dpi=600)
    plt.show()

    print(maes)
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 0.25)
    from scipy.stats import norm
    sns.distplot(np.array(maes)*100, bins=30, hist=True,
                 kde=True, ax=ax,
                 kde_kws={'linestyle': '-', 'linewidth': '0', 'color': "#4292C6"}, #'#A050A0' '#D2D2FF'
                 hist_kws={'width': 0.55, 'align': 'mid', 'color': '#BBE6FF', "edgecolor": '#000000', 'linewidth': '1.0'},
                 fit=None)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels([0, 5, 10, 15, 20],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_yticklabels(['0.00', '0.05', '0.10', '0.15', '0.20', '0.25'],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Density', labelpad=10, fontsize=22, fontweight='500')
    ax.set_xlabel('MAE of profile' + ' (%)', labelpad=10, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')
    plt.savefig('Figures/s3b.svg', dpi=600)
    plt.show()


def shap_rewrite():
    # fig = plt.figure(figsize=(9, 9))
    fig = plt.figure()
    '''these files were from /sitepackages/shap/plots/_beeswarm'''
    x_pos = np.load('Result/x_pos.npy')
    y_pos = np.load('Result/y_pos.npy')
    color = np.load('Result/data_color.npy')

    left, bottom, width, height = 0.18, 0.18, 0.7, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.set_xlim(-0.5, 0.3)
    num = 15
    ax.set_ylim(0, num+1)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xticks([-0.4, -0.2, 0.0, 0.2])
    ax.set_xticklabels(['-0.4', '-0.2', '0.0', '0.2'],
                        fontfamily='Arial', fontsize=14, fontweight='500')

    ax.set_yticks(np.arange(1, num+1, 1))
    ax.set_yticklabels(np.load('Result/features_names.npy')[-num:],
                       fontfamily='Arial', fontsize=14, fontweight='500', rotation=0)

    ax.set_ylabel('Feature', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('SHAP value', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')
    ax.axvline(x=0, zorder=-1, lw=1.0, c='grey', linestyle='dashed', alpha=1)

    colorslist = ["#FBB095", '#FCCDBC', '#FEE9E2', 'white', '#E4F4FE', '#CFEBFD', '#B2E0FC']
    colorslist.reverse()
    import matplotlib as mpl
    colormap = mpl.colors.LinearSegmentedColormap.from_list('323', colorslist, N=256, gamma=1)
    from matplotlib import cm

    for index in range(-num, 0, 1):
        x = x_pos[index]
        y = y_pos[index]
        c = color[index]
        ax.axhline(index+num+1, zorder=-1, lw=1.0, c='grey', linestyle='dashed', alpha=1)
        ax.scatter(x, y-y_pos.shape[0]+num+1, c=c, cmap=colormap, alpha=1, s=100, edgecolor='#C0C0C0', linewidth=0.000001)
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    make_axes_locatable(ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.25)

    cf = mpl.colorbar.Colorbar(cax, cmap=colormap, alpha=0.6, )
    cf.set_ticks([0, 1])
    cf.set_ticklabels(['Low', 'High'], fontfamily='Arial',
                          fontsize=14, fontweight='500', va='center')
    ax_cf = cf.ax
    ax_cf.set_ylabel('Feature value', fontfamily='Arial',
                         fontsize=16, fontweight='500', labelpad=-15)
    ax_cf.tick_params(length=4, width=1.5, which='major')
    # ax_di = cf.dividers
    cf.outline.set_linewidth(1.5)
    from PIL import Image
    plt.savefig('Figures/R2a.svg', dpi=5000)
    # plt.show()

    # png = Image.open('Figures/6a.png')
    # png.save('Figures/6a.tiff')
    # png.close()


def draw_bars(x, out_value, features, feature_type, width_separators, width_bar):
    """Draw the bars and separators."""
    rectangle_list = []
    separator_list = []

    pre_val = out_value
    for index, features in zip(range(len(features)), features):
        if feature_type == 'positive':
            bottom_bound = float(features[0])
            top_bound = pre_val
            pre_val = bottom_bound

            separator_indent = np.abs(width_separators)
            separator_pos = bottom_bound
            colors = ['#FF0D57', '#FFC3D5']
            colors = ["#FBB095", '#FEE9E2']
        else:
            bottom_bound = pre_val
            top_bound = float(features[0])
            pre_val = top_bound

            separator_indent = - np.abs(width_separators)
            separator_pos = top_bound
            colors = ['#1E88E5', '#D1E6FA']
            colors = ['#B2E0FC', '#E4F4FE']

        # Create rectangle
        if index == 0:
            if feature_type == 'positive':
                points_rectangle = [[x, bottom_bound],
                                    [x, top_bound],
                                    [width_bar+x, top_bound],
                                    [width_bar+x, bottom_bound],
                                    [(width_bar / 2)+x, bottom_bound + separator_indent]
                                    ]
            else:
                points_rectangle = [[x, bottom_bound],
                                    [x, top_bound],
                                    [(width_bar / 2)+x, top_bound + separator_indent],
                                    [width_bar+x, top_bound],
                                    [width_bar+x, bottom_bound]
                                    ]

        else:
            points_rectangle = [[x, bottom_bound],
                                [x, top_bound],
                                [(width_bar / 2)+x, top_bound + separator_indent * 0.90],
                                [width_bar+x, top_bound],
                                [width_bar+x, bottom_bound],
                                [(width_bar / 2)+x, bottom_bound + separator_indent * 0.90]
                                ]
        line = plt.Polygon(points_rectangle, closed=True, fill=True,
                           facecolor=colors[0], linewidth=0)
        # line = plt.Polygon([[0, 0], [0, 0.4], [0.5, 0.5], [0.5, 0], ], closed=True, fill=True,
        #                    facecolor=colors[0], linewidth=0)
        rectangle_list += [line]

        # Create seperator
        points_separator = [[x, separator_pos],
                            [(width_bar / 2)+x, separator_pos + separator_indent],
                            [width_bar+x, separator_pos]]

        line = plt.Polygon(points_separator, closed=None, fill=None,
                           edgecolor=colors[1], lw=2.0)
        separator_list += [line]
    return rectangle_list, separator_list


def draw_labels(fig, ax, x_pos, out_value, features, feature_type, offset_text, total_effect=0, min_perc=0.05,
                text_rotation=0):
    from matplotlib import lines
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    import matplotlib

    line_length = 1.5
    start_text = out_value
    pre_val = out_value

    # Define variables specific to positive and negative effect features
    if feature_type == 'positive':
        colors = ['#FF0D57', '#FFC3D5']
        colors = ["#EF8C49", '#FEE9E2']
        sign = 1
    else:
        colors = ['#1E88E5', '#D1E6FA']
        colors = ['#478FD1', '#E4F4FE']
        sign = -1

    # Draw initial line
    if feature_type == 'positive':
        x, y = np.array([[x_pos, x_pos+line_length], [pre_val, pre_val]])
        line = lines.Line2D(x, y, lw=1.5, alpha=0.5, color=colors[0])
        line.set_clip_on(False)
        ax.add_line(line)
        start_text = pre_val

    box_end = out_value
    val = out_value
    for feature in features:
        # Exclude all labels that do not contribute at least 10% to the total
        feature_contribution = np.abs(float(feature[0]) - pre_val) / np.abs(total_effect)
        if feature_contribution < min_perc:
            break

        # Compute value for current feature
        val = float(feature[0])

        if feature_type == 'positive':
            va = 'top'
            ratio = 0.98
        else:
            va = 'bottom'
            ratio = 1.02

        # Draw labels.
        if feature[1] == "":
            text = feature[2].split('=')[0]
        else:
            text = feature[2]

        text_out_val = plt.text(x_pos+line_length*1.2, start_text*ratio,
                                text,
                                fontsize=12, color=colors[0],
                                horizontalalignment='right',
                                va=va,
                                rotation=90)
        text_out_val.set_bbox(dict(facecolor='none', edgecolor='none'))

        # We need to draw the plot to be able to get the size of the
        # text box
        fig.canvas.draw()
        box_size = text_out_val.get_bbox_patch().get_extents() \
            .transformed(ax.transData.inverted())
        if feature_type == 'positive':
            box_end_ = box_size.get_points()[0][1]
        else:
            box_end_ = box_size.get_points()[1][1]


        # Create end line
        if (sign * box_end_) > (sign * val):
            x, y = np.array([[x_pos, x_pos + line_length], [val, val]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = val
            box_end = val

        else:
            box_end = box_end_ - sign * offset_text
            x, y = np.array([[x_pos, x_pos + line_length/2, x_pos+line_length], [val, box_end, box_end]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = box_end

        # Update previous value
        pre_val = float(feature[0])

    # Create line for labels
    extent_shading = [x_pos, x_pos+line_length,  box_end, out_value]
    path = [[x_pos, out_value], [x_pos, pre_val], [x_pos+line_length/2, box_end],
            [x_pos+line_length, box_end], [x_pos+line_length, out_value],
            [x_pos, out_value]]

    path = Path(path)
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch)


    # Create shading
    if feature_type == 'positive':
        colors = np.array([(251, 176, 179), (255, 255, 255)]) / 255.
    else:
        colors = np.array([(108, 166, 218), (255, 255, 255)]) / 255.

    cm = matplotlib.colors.LinearSegmentedColormap.from_list('cm', colors)

    _, Z2 = np.meshgrid(np.linspace(0, 10), np.linspace(-10, 10))

    im = plt.imshow(Z2.T, interpolation='quadric', cmap=cm,
                    vmax=0.01, alpha=0.8,
                    origin='lower', extent=extent_shading,
                    clip_path=patch, clip_on=True, aspect='auto')
    im.set_clip_path(patch)

    return fig, ax


def force_plot():
    data = np.load('Result/force_data_example.npy', allow_pickle=True)
    preds = [data[index][-1] for index in range(6)]

    base_values = [data[index][-2] for index in range(6)]
    total_pos = [data[index][-3] for index in range(6)]
    total_neg = [data[index][-4] for index in range(6)]
    pos_features = [data[index][-5] for index in range(6)]
    neg_features = [data[index][-6] for index in range(6)]

    fig, ax = plt.subplots(gridspec_kw={'left': 0.18, 'bottom': 0.18, 'top': 0.96, 'right': 0.96})
    ax.set_xlim(1.5, 33.5)
    ax.set_ylim(-0.05, 1.05)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(2.5, 32.5)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_xticklabels([2, 5, 9, 15, 22, 30],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.set_yticklabels(np.array(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Conc.', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('Time (min)', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')

    x_pos = [5, 10, 15, 20, 25, 30]
    threshold = 0.1
    for index in range(6):
        out_value_temp = preds[index]
        pos_features_temp = pos_features[index]
        pos_feature_type_temp = 'positive'
        width_separators = 0.01
        width_bar = 1.5
        te = np.abs(total_pos[index]) + np.abs(total_neg[index])
        rectangle_list, separator_list = draw_bars(x_pos[index]-width_bar, out_value_temp, pos_features_temp,
                                                   pos_feature_type_temp, width_separators, width_bar)
        for i in rectangle_list:
            ax.add_patch(i)
        for i in separator_list:
            ax.add_patch(i)

        offset_text = (np.abs(total_neg[index]) + np.abs(total_pos[index])) * 0.02
        draw_labels(fig, ax, x_pos[index], out_value_temp, pos_features_temp, 'positive', offset_text,
                    total_effect=te, min_perc=threshold,
                        text_rotation=0)

        neg_features_temp = neg_features[index]
        neg_feature_type_temp = 'negative'
        rectangle_list, separator_list = draw_bars(x_pos[index]-width_bar, out_value_temp, neg_features_temp,
                                                   neg_feature_type_temp, width_separators, width_bar)
        for i in rectangle_list:
            ax.add_patch(i)
        for i in separator_list:
            ax.add_patch(i)
        draw_labels(fig, ax, x_pos[index], out_value_temp, neg_features_temp, 'negative', offset_text,
                    total_effect=te, min_perc=threshold,
                    text_rotation=0)

    plt.scatter([0, 5, 10, 15, 20, 25, 30], [1]+preds, marker='*', s=100, c='#FF0D57', alpha=0.5)
    plt.scatter([28.25], [0.80], marker='*', s=100, c='#FF0D57', alpha=0.5)
    plt.text(29, 0.80, s='output', fontsize=14, color='#FF0D57', rotation=0, va='center',
             horizontalalignment='left', fontdict={'family': 'Arial', 'weight': '500'})

    for index in range(5):
        plt.vlines((index+1)*5+2.5, ax.get_ylim()[0], ax.get_ylim()[1], lw=1.5, linestyles='--', color='black')

    plt.text(29.5, 0.85, 'higher',
             fontsize=14, color='#EF8C49', rotation=90, va='bottom',
             horizontalalignment='right', fontdict={'family': 'Arial', 'weight': '500',})

    plt.text(30.5, 0.85, 'lower',
             fontsize=14, color='#478FD1', rotation=270, va='bottom',
             horizontalalignment='left', fontdict={'family': 'Arial', 'weight': '500'})

    ax.annotate("",
                xy=(29.75, 1.0),
                xytext=(29.75, 0.85),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="#EF8C49", lw=1.0))
    ax.annotate("",
                xy=(30.25, 0.85),
                xytext=(30.25, 1.0),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="#478FD1", lw=1.0))
    plt.savefig('Figures/6b.svg', dpi=600)
    plt.show()


def baseline_des_alg():
    data = np.load('Result/metrics_base_dess_alg.npy')
    data = np.mean(np.mean(data, axis=0), axis=1)
    mask_odds = [index for index in range(data.shape[0]) if index % 2 == 1]
    data = data[mask_odds, 0].reshape((10, 8))*100

    fps = ['rdk', 'mor', '3dm', 'whi', 'rdk+whi', 'mor+whi', 'rdk+3dm', 'mor+3dm']
    models_name = ['xgb', 'svr', 'kr', 'kn', 'bag', 'dt', 'mlp', 'rf', 'et', 'ada']
    mask_adjust = [1, 5, 2, 4, 3, 0, 6, 7, 9, 8]
    data = data[mask_adjust, :].T
    models_name = np.array(models_name)[[mask_adjust]]

    colorlist = palettable.colorbrewer.sequential.Blues_9.hex_colors[1:-3]
    colorlist.reverse()
    colormap = mpl.colors.LinearSegmentedColormap.from_list('323', colorlist, N=256, gamma=1.0)
    norm = mpl.colors.Normalize(vmax=data.max(), vmin=data.min())
    value2color = lambda x: colormap(norm(x))
    colors = np.array([value2color(cor) for cor in data.reshape(1, -1).tolist()]) \
        .reshape(data.shape[0], data.shape[1], 4)
    fill_colors = colors

    x_cof = np.round(data, 2)
    n = x_cof.shape[0]
    m = x_cof.shape[1]

    fig = plt.figure()
    gs = gridspec.GridSpec(n, m)
    gs.update(wspace=0.0, hspace=0.0, left=0.18, bottom=0.18, right=0.8, top=0.96)

    for i, j in product(range(0, n, 1), range(0, m, 1)):
        if i == 5 and j == 5:
            ax = plt.subplot(gs[i, j], zorder=1)
        else:
            ax = plt.subplot(gs[i, j], zorder=0)
        ax.set_facecolor(fill_colors[i, j])
        [ax.spines[_].set_color('w') for _ in ['right', 'top', 'left', 'bottom']]
        ax.text(0.5, 0.5, x_cof[i, j],
                    fontdict={'family': 'Arial', 'weight': '500', 'color': '#FFFFFF'},  # args
                    fontsize=10,  # c_arg
                    horizontalalignment='center', verticalalignment='center_baseline', c='black')
        if j == 0:
            ax.text(-0.2, 0.5, s=fps[i], horizontalalignment='right', verticalalignment='center_baseline',
                    fontdict={'family': 'Arial', 'size': '14', 'weight': '500'})
        if i == 7:
            ax.text(0.5, -0.2, s=models_name[j], horizontalalignment='center', verticalalignment='top',
                    fontdict={'family': 'Arial', 'size': '14', 'weight': '500'})

        ax.set_xticks([])
        ax.set_yticks([])
        if i == 5 and j == 5:
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['right'].set_linewidth(1.0)
            ax.spines['top'].set_linewidth(1.0)
            ax.spines['bottom'].set_linewidth(1.0)

            ax.spines['left'].set_color('#C00000')
            ax.spines['right'].set_color('#C00000')
            ax.spines['top'].set_color('#C00000')
            ax.spines['bottom'].set_color('#C00000')

    cbar_ax = fig.add_axes([0.82, 0.18, 0.03, 0.78])
    ColorbarBase(cbar_ax, cmap=colormap, norm=norm)
    ColorbarBase(cbar_ax, cmap=colormap, norm=norm)
    cf = ColorbarBase(cbar_ax, cmap=colormap, norm=norm)
    ax_cf = cf.ax
    ax_cf.tick_params(length=4, width=1.5, which='major')
    # ax_cf.set_ylim(0.46, 1.1)
    ax_cf.set_yticks([6, 7, 8, 9, 10, 11, 12, 13, 14])
    ax_cf.set_yticklabels([6, 7, 8, 9, 10, 11, 12, 13, 14],
                          fontfamily='Arial', fontsize=12, fontweight='500')
    plt.text(x=4, y=10, s='MAE of Conc. (%)',
             rotation='vertical', fontfamily='Arial',
             fontsize=14, fontweight='500', va='center', ha='center')
    cf.outline.set_linewidth(1.5)
    plt.savefig('Figures/s1a.svg', dpi=600)
    plt.show()


def baseline_lags():
    data = np.load('Result/metrics_base_lags.npy', allow_pickle=True)[:, :, :, 0]
    data = np.concatenate(data, axis=1)

    mean = np.mean(data, axis=1)[0:6]*100
    std = np.std(data, axis=1)[0:6]*100
    lags = [0, 1, 2, 3, 4, 5]

    plt.figure()
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(-0.7, 5.7)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels([0, 1, 2, 3, 4, 5],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylim(2, 7)
    ax.set_yticks(np.array([2, 3, 4, 5, 6, 7]))
    ax.set_yticklabels(np.array([2, 3, 4, 5, 6, 7]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('MAE of Conc.' + ' (%)', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('Order of lagged variable', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')

    error_attri = dict(elinewidth=1.0, ecolor="black", capsize=5)
    ax.bar(lags, mean, color="#FCEDE2", width=0.6, align="center",
            yerr=std, error_kw=error_attri, linewidth=1.0, edgecolor='black') # #DFEBF6
    plt.savefig('Figures/s1b.svg', dpi=600)
    plt.show()


def baseline_multis():
    data = np.load('Result/metrics_base_shifts.npy', allow_pickle=True)[:, :, :, 0]
    data = np.concatenate(data, axis=1)
    print(data.shape)

    mean = np.mean(data, axis=1)[0:6]*100
    std = np.std(data, axis=1)[0:6]*100
    lags = [1, 2, 3, 4, 5, 6]

    plt.figure()
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(0.3, 6.7)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([1, 2, 3, 4, 5, 6],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylim(2, 6)
    ax.set_yticks(np.array([2, 3, 4, 5, 6]))
    ax.set_yticklabels(np.array([2, 3, 4, 5, 6]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('MAE of Conc.' + ' (%)', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('Multiple Estimation', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')

    error_attri = dict(elinewidth=1.0, ecolor="black", capsize=5)
    ax.bar(lags, mean, color="#D8EDD9", width=0.6, align="center",
            yerr=std, error_kw=error_attri, linewidth=1.0, edgecolor='black')
    plt.savefig('Figures/s1c.svg', dpi=600)
    plt.show()


def feature_contirbution_plot():
    metric = np.load('Result/metric_contribution.npy')[:, :, 0]
    means = []
    stds = []

    for index in range(metric.shape[0]):
        metric_attr = metric[index, :]
        mean = np.mean(metric_attr, axis=0)
        std = np.std(metric_attr, axis=0)
        means.append(mean)
        stds.append(std)

    plt.figure()
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(-0.5, 9.5)
    ax.set_xticks(np.arange(0, metric.shape[0], 1))
    ax.set_xticklabels(['top05', 'top10', 'top15', 'top20', 'top25', 'top30', 'top35', 'top40', 'top45', 'top50'],
                       fontfamily='Arial', fontsize=20, fontweight='500', rotation=30)
    ax.set_ylim(3, 11)
    ax.set_yticks(np.arange(4, 11, 2))
    ax.set_yticklabels(np.arange(4, 11, 2),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('MAE of Conc.' + ' (%)', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel('', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')

    color = '#B9DCFF'
    line_color = "#4292C6"
    x = np.arange(0, metric.shape[0], 1)
    plt.fill_between(x, (np.array(means)-np.array(stds))*100, (np.array(means)+np.array(stds))*100, alpha=0.3,
                     facecolor=color)
    plt.plot(x, np.array(means)*100, c=line_color, marker='o', markersize=6, linewidth=1.5, alpha=0.8,)
    plt.savefig('Figures/s4.svg', dpi=600)
    plt.show()
