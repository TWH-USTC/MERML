import pandas as pd
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.metrics import r2_score
from itertools import product
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import ColorbarBase
import palettable

mpl.rcParams["mathtext.fontset"] = 'custom'
mpl.rcParams["mathtext.bf"] = "Arial:bold"
mpl.rcParams["mathtext.default"] = 'regular'


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def equal_array(array_1, array_2):
    # assert np.all(array_1.shape == array_2.shape)
    return np.all(array_1 == array_2)


def in_list(array_attr, list_attr):
    flag = False
    for element in list_attr:
        if equal_array(array_attr, element):
            flag = True
            break
        else:
            pass
    return flag


def unique_array(data):
    result = []
    for index in range(data.shape[0]):
        try:
            element = data[index, :]
        except IndexError:
            element = data[index]
        if len(result) == 0:
            result.append(element)
        else:
            if in_list(element, result):
                continue
            else:
                result.append(element)
    return result


def average_rate(condition, data):
    conversion_under_condition = []
    for index in range(data.shape[0]):
        element = data[index, :]
        if equal_array(element[0:3], condition):
            conversion_under_condition.append(element[-1])
        else:
            pass
    if len(conversion_under_condition) >= 12:
        return conversion_under_condition
    else:
        return 0


def condition_plot(data):
    conditions = data[:, -10:-7]
    compound_conc = unique_array(conditions[:, 0])
    condition_unique = unique_array(conditions)
    conversion_list = []
    for condition in condition_unique:
        average_conversion = average_rate(condition, data[:, -10:])
        if isinstance(average_conversion, int):
            conversion_list.append(0)
        else:
            conversion_list.append(np.average(average_conversion))
    data_plot = [[] for index in range(len(compound_conc))]
    for index in range(len(condition_unique)):
        position = compound_conc.index(condition_unique[index][0])
        if conversion_list[index] != 0:
            data_plot[position].append([condition_unique[index][1], condition_unique[index][2], conversion_list[index]])
        else:
            pass

    # c = ['#BBE6FF', '#D8EDD9', '#FCEDE2']
    c = ['#9DC3E6', '#F4B183', '#B39BC7']
    for index in range(0, 3, 1):
        data_norm = np.array(data_plot[index])[:, 2]
        from sklearn.preprocessing import MinMaxScaler
        data_norm = MinMaxScaler().fit_transform(data_norm.reshape(-1, 1))

        plt.figure()
        left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
        rect = [left, bottom, width, height]
        ax = plt.axes(rect)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        mpl.rcParams["mathtext.fontset"] = 'custom'
        mpl.rcParams["mathtext.bf"] = "Arial:bold"
        mpl.rcParams["mathtext.default"] = 'regular'

        ax.set_ylabel('Conc. of Fe(II)', fontfamily='Arial', fontsize=22, fontweight='500')
        ax.set_xlabel('Conc. of ' + r'$H_2O_2$', fontfamily='Arial', fontsize=22, fontweight='500')
        ax.tick_params(length=4, width=1.5, which='major')

        mpl.rcParams["markers.fillstyle"] = 'full'
        for index_point in range(len(data_plot[index])):
            point = data_plot[index][index_point]
            print([point[0], point[1], 1 - data_norm[index_point][0]])
            plt.scatter(point[1], point[0], marker='o', s=((data_norm[index_point]) * 300 + 50), c=c[index],
                        alpha=0.8, zorder=1, edgecolor=None, lw=0)

        if index == 0:
            ax.set_xlim(4.5, 13.5)
            ax.set_ylim(-0.5, 10.5)
            ax.set_xticks([5, 7, 9, 11, 13])
            ax.set_yticks([0, 2, 4, 6, 8, 10])
            ax.set_xticklabels([5, 7, 9, 11, 13], fontfamily='Arial', fontsize=18,
                               fontweight='500')
            ax.set_yticklabels([0, 2, 4, 6, 8, 10], fontfamily='Arial', fontsize=18,
                               fontweight='500')
            mpl.rcParams["markers.fillstyle"] = 'none'
            ax.scatter(10, 5, c='#C26275', s=400, lw=2)
            ax.scatter(9.5, 9.5, c='#C26275', s=200, lw=2)
            ax.text(x=10, y=9.5, s='Optimized ratio',
                    fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va':'center_baseline'})
            # plt.savefig('Figures/s5a.svg', dpi=600)
            plt.show()
        elif index == 1:
            ax.set_xlim(13, 57)
            ax.set_ylim(4, 21)
            ax.set_xticks([15, 25, 35, 45, 55])
            ax.set_yticks([5, 10, 15, 20])
            ax.set_xticklabels([15, 25, 35, 45, 55], fontfamily='Arial', fontsize=18,
                               fontweight='500')
            ax.set_yticklabels([5, 10, 15, 20], fontfamily='Arial', fontsize=18,
                               fontweight='500')
            mpl.rcParams["markers.fillstyle"] = 'none'
            ax.scatter(40, 10, c='#C26275', s=400, lw=2)
            ax.scatter(37.5, 19.5, c='#C26275', s=200, lw=2)
            ax.text(x=40, y=19.5, s='Optimized ratio',
                    fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline'})
            plt.show()
        elif index == 2:
            ax.set_xlim(28, 72)
            ax.set_ylim(4, 21)
            ax.set_xticks([30, 40, 50, 60, 70])
            ax.set_yticks([5, 10, 15, 20])
            ax.set_xticklabels([30, 40, 50, 60, 70], fontfamily='Arial', fontsize=18,
                               fontweight='500')
            ax.set_yticklabels([5, 10, 15, 20], fontfamily='Arial', fontsize=18,
                               fontweight='500')
            mpl.rcParams["markers.fillstyle"] = 'none'
            ax.scatter(70, 10, c='#C26275', s=400, lw=2)
            ax.scatter(30, 6, c='#C26275', s=200, lw=2)
            ax.text(x=32.5, y=6, s='Optimized ratio',
                    fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline'})
            plt.show()
        plt.show()


def violin(data):
    name = np.array(unique_array(data[:, 0]))
    conversion = data[:, -6].reshape(-1, 12).astype(np.float32)
    # for index in range(1, 7, 1):
    #     if index == 1:
    #         conversion = data[:, -1].reshape(-1, 12).astype(np.float32)
    #     else:
    #         conversion = np.r_[conversion, data[:, -index].reshape(-1, 12).astype(np.float32)]


    result = plt.violinplot(conversion, positions=np.arange(0, 12, 1), showmedians=True)
    cmedians = result['cmedians'].get_segments()
    c_medians = []
    for element in cmedians:
        c_medians.append(element[0][1])
    mask = np.argsort(c_medians)
    mask = [4, 7, 9, 11, 0, 1, 5, 6, 8, 10, 3, 2]
    c_medians = np.array(c_medians)[mask]
    conversion = conversion[:, mask]
    name = name[mask]
    print(name)

    plt.figure()
    left, bottom, width, height = 0.18, 0.38, 0.78, 0.58
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['top'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    ax.spines['left'].set_color('#000000')
    ax.spines['right'].set_color('#000000')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # 绘制小提琴图
    kws = {'linewidth':'0.5', 'color':'#000000'}
    plt.hlines(y=1-np.average(conversion.reshape(1, -1)[0, :]), xmin=-0.5, xmax=11.5, linestyles='--', color='grey', lw=2)
    plt.scatter(np.arange(0, 12, 1), 1 - c_medians, zorder=3, marker='*', s=200, c='#C26275', edgecolor="#F8BC31", lw=0.7)
    plt.scatter(7.5, 0.925, marker='*', s=200, c='#C26275', edgecolor="#F8BC31", lw=0.7)
    plt.text(8, 0.925, s='mean value',
             fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline'})
    plt.boxplot(positions=np.arange(0, 12, 1), x= 1 - conversion, zorder=2, widths=0.2, showfliers=False,
                boxprops=kws, whiskerprops=kws,
                capprops=kws, medianprops=kws)

    # plt.hlines(y=0.63, xmin=4.25, xmax=4.85, linestyles='--', color='grey', lw=2)
    # plt.text(5, 0.635, s='mean value of all data',
    #          fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline'})

    plt.hlines(y=0.13, xmin=0.25, xmax=0.85, linestyles='--', color='grey', lw=2)
    plt.text(1, 0.135, s='mean value of all data',
             fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline'})

    color = ["#C8557C", "#A050A0", "#4682B4", "#5F9EA0", "#E9967A", "#F5DEB3", "#663366",
             "#CC3366", "#DB7093", "#CC0000",
             "#336699", "#99CCFF", "#0066CC", "#336666", "#5F9EA0", "#66CDAA", "#FF9900", "#F5DEB3"]
    ax = sns.violinplot(data=1-conversion, bw=0.3, cut=0, linewidth=0, orient="v",
                   scale='area', inner=None, zorder=1, width=0.6, color='#CDE0EE', saturation=1)

    rect = mpl.patches.Rectangle(xy=(-0.5, -0.05), width=6, height=1.05, facecolor=color[3], edgecolor='black',
                                 linewidth=0, alpha=0.15, zorder=-1)
    ax.add_patch(rect)
    rect = mpl.patches.Rectangle(xy=(5.5, -0.05), width=6, height=1.05, facecolor=color[4], edgecolor='black',
                                 linewidth=0, alpha=0.15, zorder=-1)
    ax.add_patch(rect)

    # ax.text(2.5, 0.8, s='electron donating',
    #         fontdict={'fontfamily': 'Arial', 'fontsize': '16', 'fontweight': '500', 'va': 'center_baseline', 'ha':'center'})
    # ax.annotate(text='', xy=(-0.5, 0.75), xytext=(5.5, 0.75), arrowprops={'arrowstyle':'->', 'mutation_aspect':1.5})
    # ax.text(8.5, 0.8, s='electron withdrawing',
    #         fontdict={'fontfamily': 'Arial', 'fontsize': '16', 'fontweight': '500', 'va': 'center_baseline',
    #                   'ha': 'center'})
    # ax.annotate(text='', xy=(5.5, 0.75), xytext=(11.5, 0.75), arrowprops={'arrowstyle': '<-', 'mutation_aspect': 1.5})


    ax.text(2.5, 0.05, s='electron donating',
            fontdict={'fontfamily': 'Arial', 'fontsize': '16', 'fontweight': '500', 'va': 'center_baseline',
                      'ha': 'center'})
    ax.annotate(text='', xy=(-0.5, 0), xytext=(5.5, 0), arrowprops={'arrowstyle': '->', 'mutation_aspect': 1.5})
    ax.text(8.5, 0.05, s='electron withdrawing',
            fontdict={'fontfamily': 'Arial', 'fontsize': '16', 'fontweight': '500', 'va': 'center_baseline',
                      'ha': 'center'})
    ax.annotate(text='', xy=(5.5, 0), xytext=(11.5, 0), arrowprops={'arrowstyle': '<-', 'mutation_aspect': 1.5})

    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['R-OH', r'$R-OCH_3$', r'$R-CH_2OH$', r'$R-CH_3$', r'$R-H$', r'$R-Cl$', 'R-CHO',
                        r'$R-COCH_3$', r'R-COOR', r'$R-CONH_2$', 'R-COOH', r'$R-NO_2$'], fontfamily='Arial', fontsize=18,
                       fontweight='500', rotation=90, ha='center')
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],fontfamily='Arial',
                       fontsize=18, fontweight='500')
    ax.set_ylabel('Conc.',fontfamily='Arial', fontsize=22, fontweight='500')
    ax.set_xlabel('Type of compounds',fontfamily='Arial', fontsize=22, fontweight='500')
    ax.tick_params(length=4, width=1.5, which='major')
    plt.show()


def different_order_ort(data):
    mask_1 = [3 + 12*i for i in range(25)]
    mask_2 = [5 + 12 * i for i in range(25)]
    mask_3 = [9 + 12 * i for i in range(25)]
    mask_4 = [10 + 12*i for i in range(25)]
    label = ['p-hydroxybenzoic acid', 'p-hydroxybenzaldehyde',
             'p-hydroxybenzyl alcohol', 'p-acetamidophenol']

    data_1 = 1 - data[mask_1, -6]
    data_2 = 1 - data[mask_2, -6]
    data_3 = 1 - data[mask_3, -6]
    data_4 = 1 - data[mask_4, -6]
    plt.figure()
    left, bottom, width, height = 0.2, 0.18, 0.7, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    # 绘制小提琴图

    mpl.rcParams["markers.fillstyle"] = 'none'
    color = ["#C8557C", "#A050A0", "#4682B4", "#5F9EA0", "#E9967A", "#F5DEB3", "#663366",
             "#CC3366", "#DB7093", "#CC0000",
             "#336699", "#99CCFF", "#0066CC", "#336666", "#5F9EA0", "#66CDAA", "#FF9900", "#F5DEB3"]
    for index in range(0, 25, 1):
        conversions = [data_1[index], data_2[index], data_3[index], data_4[index]]
        conversion_max = min(conversions)
        mpl.rcParams["markers.fillstyle"] = 'none'
        plt.scatter([index+1 for i in range(4)], conversions, zorder=0, marker='o', s=120, c=color[0:4], lw=2, alpha=0.6)
        mpl.rcParams["markers.fillstyle"] = 'full'
        plt.scatter(index+1, conversion_max, zorder=0, marker='o', s=80, c='#F8BC31', lw=0, alpha=0.6)
        ax.axvline(index+1, linestyle='--', c='grey', lw=0.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.set_xlim(0, 26)
    ax.set_xticks(np.arange(0, 26, 5).astype(int))

    ax.set_xticklabels(np.arange(0, 26, 5).astype(int), fontfamily='Arial', fontsize=18,
                       fontweight='500')
    ax.set_xlabel('Reaction condition index', fontfamily='Arial', fontsize=22, fontweight='500')

    ax.set_ylim(-0.2, 1.0)
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['-0.2', '0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontfamily='Arial', fontsize=18, fontweight='500')

    ax.set_ylabel('Conc.', fontfamily='Arial', fontsize=22, fontweight='500')

    ax.tick_params(length=4, width=1.5, which='major')

    pos_x = 1
    pos_y = 0.20
    shift_x = 1
    shift_y = 0.08
    ax.scatter(1, pos_y, zorder=0, marker='o', s=100, c='#F8BC31', lw=0, alpha=0.6)
    ax.text(x=2, y=pos_y, s='fast reaction',
            fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline'})
    mpl.rcParams["markers.fillstyle"] = 'none'
    ax.scatter(1, pos_y-shift_y*2, zorder=0, marker='o', s=100, c=color[0], lw=2, alpha=0.6)
    ax.text(x=2, y=pos_y-shift_y*2, s=label[0],
            fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline'})
    ax.scatter(1, pos_y-shift_y*3, zorder=0, marker='o', s=100, c=color[1], lw=2, alpha=0.6)
    ax.text(x=2, y=pos_y-shift_y*3, s=label[1],
            fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline'})
    ax.scatter(1, pos_y-shift_y*4, zorder=0, marker='o', s=100, c=color[2], lw=2, alpha=0.6)
    ax.text(x=2, y=pos_y-shift_y*4, s=label[2],
            fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline'})
    ax.scatter(1, pos_y-shift_y, zorder=0, marker='o', s=100, c=color[3], lw=2, alpha=0.6)
    ax.text(x=2, y=pos_y-shift_y, s=label[3],
            fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline'})
    plt.show()


def get_slope(data):
    curve = data[:, -7:]
    slope = copy.copy(data[:, 1:])
    slope[:, 0] = (1 - curve[:, 1]) / 2
    slope[:, 1] = (curve[:, 1] - curve[:, 2]) / 3
    slope[:, 2] = (curve[:, 2] - curve[:, 3]) / 4
    slope[:, 3] = (curve[:, 3] - curve[:, 4]) / 6
    slope[:, 4] = (curve[:, 4] - curve[:, 5]) / 7
    slope[:, 5] = (curve[:, 5] - curve[:, 6]) / 8
    return slope


def get_cluster_label(slope_attr, n_cluster):
    scale = MinMaxScaler(feature_range=(0, 1))
    scale.fit(slope_attr)
    slope = scale.transform(slope_attr)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    kmeans.fit(slope)
    clusterid = kmeans.predict(slope)
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(slope, clusterid)

    # from Bio.Cluster import kcluster
    # clusterid, error, npass = kcluster(slope, nclusters=n_cluster, dist='u')
    # from sklearn.metrics import silhouette_score
    # silhouette_avg = silhouette_score(slope, clusterid, metric='cosine')
    print('silhouette_avg: ', end='')
    print(silhouette_avg)

    list_clusters = [[] for i in range(n_cluster)]
    for index in range(len(clusterid)):
        category = clusterid[index]
        list_clusters[category].append(slope[index, :].tolist())
    centers = []
    print(list_clusters)
    for element in list_clusters:
        centers.append(np.average(element, axis=0))
    np.savetxt('cluster_result.txt', clusterid, delimiter=',', fmt='%s')

    centers_inverse = scale.inverse_transform(np.array(centers))
    print(centers_inverse)
    curves_centers = []
    for element in centers_inverse:
        ratio_2 = 1 - 2*element[0]
        ratio_5 = (ratio_2 - 3*element[1])
        ratio_9 = (ratio_5 - 4* element[2])
        ratio_15 = (ratio_9 - 6* element[3])
        ratio_22 = (ratio_15 - 7* element[4])
        ratio_30 = (ratio_22 - 8* element[5])
        curve_temp = [1, ratio_2, ratio_5, ratio_9, ratio_15, ratio_22, ratio_30]
        curves_centers.append(curve_temp)
    print(curves_centers)
    return clusterid, centers, curves_centers


def cluster_plot(data, n_cluster):
    curve_slope = get_slope(data)
    cluster_label, cluster_centers, curve_centers = get_cluster_label(curve_slope, n_cluster)
    np.savetxt('cluster_label.txt', cluster_label)
    scale_mm = MinMaxScaler(feature_range=(0, 1))
    scale_mm.fit(curve_slope)
    curve_slope = scale_mm.transform(curve_slope)
    scale = PCA(n_components=2)
    pca = scale.fit(curve_slope)
    slope = pca.transform(curve_slope)
    centers = pca.transform(cluster_centers)
    variance = pca.explained_variance_ratio_
    dimension_origin = 1 - np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    dimension_origin = scale_mm.transform(dimension_origin)
    dimension_origin_pca = pca.transform(dimension_origin)
    print(dimension_origin_pca)
    for index in range(dimension_origin_pca.shape[0]):
        dimension_origin_pca[index, :] = normalize(dimension_origin_pca[index, :])

    color = ["#C8557C", "#A050A0", "#4682B4", "#5F9EA0", "#E9967A", "#F5DEB3","#663366",
             "#CC3366", "#DB7093", "#CC0000",
             "#336699", "#99CCFF", "#0066CC", "#336666", "#5F9EA0", "#66CDAA", "#FF9900", "#F5DEB3"]
    # pca
    import matplotlib as mpl
    mpl.rcParams["mathtext.fontset"] = 'custom'
    mpl.rcParams["mathtext.bf"] = "Arial:bold"
    mpl.rcParams["mathtext.default"] = 'regular'

    plt.figure()
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['top'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    ax.spines['left'].set_color('#000000')
    ax.spines['right'].set_color('#000000')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.set_ylim(-0.75, 0.75)
    ax.set_xlim(-0.75, 1.25)
    ax.set_xlabel('PCA 1 (%s' %("{:.1f}".format(variance[0]*100)) + '%)', fontfamily='Arial', fontsize=22, fontweight='500', labelpad=5)
    ax.set_ylabel('PCA 2 (%s' %("{:.1f}".format(variance[1]*100)) + '%)', fontfamily='Arial', fontsize=22, fontweight='500', labelpad=5)
    ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels([-0.5, 0.0, 0.5, 1.0],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    ax.set_yticklabels([-0.6, -0.3, 0.0, 0.3, 0.6],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.tick_params(length=4, width=1.5, which='major')
    ax.axhline(0.0, linestyle='--', c='#4D4D4D', lw=1.5)
    ax.axvline(0.0, linestyle='--', c='#4D4D4D', lw=1.5)
    ax.scatter(-0.6, -0.625, marker='*', s=200, c='#C26275', zorder=2, edgecolor='#000000', lw=0.5)
    ax.text(-0.5, -0.625, s='cluster center', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline',
                      'ha': 'left'})
    for index in range(slope.shape[0]):
        plt.scatter(slope[index, 0], slope[index, 1], c=color[int(cluster_label[index])], alpha=0.6)
    # c='#C26275', edgecolor='#A61B29', lw=1.5
    # for index in range(len(dimension_origin_pca)):
    #     end = dimension_origin_pca[index]
    #     plt.arrow(0, 0, end[0]*0.5, end[1]*0.5, head_width=0.03, lw=1, length_includes_head=True, head_length=0.045, fc='#4D4D4D',
    #               ec='#4D4D4D', zorder=1)
    for index in range(len(centers)):
        element = centers[index]
        plt.scatter(element[0], element[1], marker='*', s=200, c='#C26275', zorder=2, edgecolor='#000000', lw=0.5)
    plt.show()

    plt.figure()
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-1, 31)
    ax.set_xlabel('Time (min)', fontfamily='Arial', fontsize=22, fontweight='500', labelpad=5)
    ax.set_ylabel('Conc.', fontfamily='Arial', fontsize=22, fontweight='500', labelpad=5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.tick_params(length=4, width=1.5, which='major')
    for index in range(len(curve_centers)):
        element = curve_centers[index]
        plt.plot([0, 2, 5, 9, 15, 22, 30], element, c=color[index], marker='D', markersize=7, alpha=0.8)
    plt.show()


def cluster_plot_curve(data, n_cluster):
    curve_slope = get_slope(data)
    cluster_label, cluster_centers, curve_centers = get_cluster_label(curve_slope, n_cluster)

    data = data[:, 1:]
    print(data)
    # scale_mm = MinMaxScaler(feature_range=(0, 1))
    # scale_mm.fit(data[:, 1:])
    # data = scale_mm.transform(data[:, 1:])
    scale = PCA(n_components=2)
    pca = scale.fit(data)
    data = pca.transform(data)

    centers = pca.transform(cluster_centers)
    variance = pca.explained_variance_ratio_
    dimension_origin = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    # dimension_origin = scale_mm.transform(dimension_origin)
    dimension_origin_pca = pca.transform(dimension_origin)
    print(dimension_origin_pca)
    for index in range(dimension_origin_pca.shape[0]):
        dimension_origin_pca[index, :] = normalize(dimension_origin_pca[index, :])

    color = ["#C8557C", "#A050A0", "#4682B4", "#5F9EA0", "#E9967A", "#F5DEB3", "#663366",
             "#CC3366", "#DB7093", "#CC0000",
             "#336699", "#99CCFF", "#0066CC", "#336666", "#5F9EA0", "#66CDAA", "#FF9900", "#F5DEB3"]

    import matplotlib as mpl
    mpl.rcParams["mathtext.fontset"] = 'custom'
    mpl.rcParams["mathtext.bf"] = "Arial:bold"
    mpl.rcParams["mathtext.default"] = 'regular'

    plt.figure()
    left, bottom, width, height = 0.2, 0.18, 0.70, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    # ax.set_ylim(-0.75, 0.75)
    # ax.set_xlim(-0.75, 1.25)
    ax.set_xlabel('PC1 (%s' % ("{:.1f}".format(variance[0] * 100)) + '%)', fontfamily='Arial', fontsize=28,
                  fontweight='500', labelpad=5)
    ax.set_ylabel('PC2 (%s' % ("{:.1f}".format(variance[1] * 100)) + '%)', fontfamily='Arial', fontsize=28,
                  fontweight='500', labelpad=5)
    # ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
    # ax.set_xticklabels([-0.5, 0.0, 0.5, 1.0],
    #                    fontfamily='Arial', fontsize=22, fontweight='500')
    # ax.set_yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    # ax.set_yticklabels([-0.6, -0.3, 0.0, 0.3, 0.6],
    #                    fontfamily='Arial', fontsize=22, fontweight='500')
    ax.tick_params(length=4, width=1.5, which='major')
    ax.axhline(0.0, linestyle='--', c='#4D4D4D', lw=1.5)
    ax.axvline(0.0, linestyle='--', c='#4D4D4D', lw=1.5)
    for index in range(data.shape[0]):
        plt.scatter(data[index, 0], data[index, 1], c=color[int(cluster_label[index])], alpha=0.6)
    # c='#C26275', edgecolor='#A61B29', lw=1.5
    for index in range(len(dimension_origin_pca)):
        end = dimension_origin_pca[index]
        plt.arrow(0, 0, end[0] * 0.5, end[1] * 0.5, head_width=0.03, lw=1, length_includes_head=True, head_length=0.045,
                  fc='#4D4D4D',
                  ec='#4D4D4D', zorder=1)
    for index in range(len(centers)):
        element = centers[index]
        plt.scatter(element[0], element[1], marker='*', s=200, c='#C26275', zorder=2, edgecolor='#000000', lw=0.5)
    plt.show()

    plt.figure()
    left, bottom, width, height = 0.2, 0.18, 0.70, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, 30.5)
    ax.set_xlabel('Time (min)', fontfamily='Arial', fontsize=28, fontweight='500', labelpad=5)
    ax.set_ylabel('Conversion', fontfamily='Arial', fontsize=28, fontweight='500', labelpad=5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30],
                       fontfamily='Arial', fontsize=22, fontweight='500')
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       fontfamily='Arial', fontsize=22, fontweight='500')
    ax.tick_params(length=4, width=1.5, which='major')
    for index in range(len(curve_centers)):
        element = curve_centers[index]
        plt.plot([0, 2, 5, 9, 15, 22, 30], element, c=color[index], marker='D', markersize=7, alpha=0.8)
    plt.show()


def figure_si_2(data):
    x = [0, 2, 5, 9, 15, 22, 30]
    y_11 = data[227-12, -7:]
    y_12 = data[251-12, -7:]
    y_13 = data[239-12, -7:]
    y_14 = data[263-12, -7:]

    y_21 = data[193 - 12, -7:]
    y_22 = data[194 - 12, -7:]
    y_23 = data[198 - 12, -7:]

    y_31 = data[223 - 12, -7:]
    y_32 = data[235 - 12, -7:]
    y_33 = data[295 - 12, -7:]

    color = ["#C8557C", "#A050A0", "#4682B4", "#5F9EA0", "#E9967A", "#F5DEB3", "#663366",
             "#CC3366", "#DB7093", "#CC0000",
             "#336699", "#99CCFF", "#0066CC", "#336666", "#5F9EA0", "#66CDAA", "#FF9900", "#F5DEB3"]
    import matplotlib as mpl
    mpl.rcParams["mathtext.fontset"] = 'custom'
    mpl.rcParams["mathtext.bf"] = "Arial:bold"
    mpl.rcParams["mathtext.default"] = 'regular'

    plt.figure()
    left, bottom, width, height = 0.2, 0.18, 0.70, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, 30.5)
    ax.set_xlabel('Time (min)', fontfamily='Arial', fontsize=28, fontweight='500', labelpad=5)
    ax.set_ylabel('Conversion', fontfamily='Arial', fontsize=28, fontweight='500', labelpad=5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30],
                       fontfamily='Arial', fontsize=22, fontweight='500')
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       fontfamily='Arial', fontsize=22, fontweight='500')
    ax.tick_params(length=4, width=1.5, which='major')
    # plt.plot(x, y_11, c=color[2], marker='D', markersize=7, alpha=0.8, label='10/30 mg/L')
    # plt.plot(x, y_12, c=color[3], marker='o', markersize=7, alpha=0.8, label='20/30 mg/L')

    # plt.plot(x, y_13, c=color[2], marker='D', markersize=7, alpha=0.8, label='10/50 mg/L')
    # plt.plot(x, y_14, c=color[3], marker='o', markersize=7, alpha=0.8, label='15/50 mg/L')

    # plt.plot(x, y_21, c=color[1], marker='D', markersize=7, alpha=0.8, label='4-CP')
    # plt.plot(x, y_22, c=color[2], marker='D', markersize=7, alpha=0.8, label='4-NP')
    # plt.plot(x, y_23, c=color[3], marker='D', markersize=7, alpha=0.8, label='4-HAP')

    plt.plot(x, y_31, c=color[1], marker='D', markersize=7, alpha=0.8, label='10/30 mg/L')
    plt.plot(x, y_32, c=color[2], marker='D', markersize=7, alpha=0.8, label='10/50 mg/L')
    plt.plot(x, y_33, c=color[3], marker='D', markersize=7, alpha=0.8, label='10/70 mg/L')

    plt.legend(fontsize=16, ncol=1)
    plt.show()


def f_1(X, k):
    return k*X


def get_rate_constant(curves):
    number_sample, number_feature = curves.shape
    k_list = []
    r2_list = []
    for index in range(number_sample):
        curves_temp = curves[index, :].tolist()
        x_0 = [0, 2, 5, 9, 15, 22, 30]
        for index in range(len(curves_temp)):
            if np.isinf(curves_temp[index]):
                curves_temp = curves_temp[0:index]
                x_0 = x_0[0:index]
                break

        k = optimize.curve_fit(f_1, x_0, curves_temp)[0]
        func_k = lambda x: f_1(x, k)
        y_pred = list(map(func_k, x_0))
        r2_list.append(r2_score(curves_temp, y_pred))
        k_list.append(k[0])
    return k_list, r2_list


def zero_order(curves_attr):
    curves = copy.copy(curves_attr)
    return 1 - curves


def first_order(curves_attr):
    curves = copy.copy(curves_attr)
    for index_row in range(curves.shape[0]):
        for index_col in range(curves.shape[1]):
            curves[index_row][index_col] = - np.log(1 - curves[index_row][index_col])
    return curves


def second_order(curves_attr):
    curves = copy.copy(curves_attr)
    for index_row in range(curves.shape[0]):
        for index_col in range(curves.shape[1]):
            try:
                curves[index_row][index_col] = 1/(1 - curves[index_row][index_col]) - 1
            except:
                curves[index_row][index_col] = np.inf
    return curves


def r2(actual, pred):
    average = np.average(actual)
    func_sst = lambda x : (x-average)**2
    sst = np.sum(list(map(func_sst, actual)))
    func_ssr = lambda x,y : (x-y)**2
    ssr = np.sum(list(map(func_ssr, actual, pred)))
    return 1 - ssr/sst


def corr_plot(x_cof, name, types='circle', front_raito=1, label_axis="off"):
    colorlist = palettable.cartocolors.diverging.Earth_7.mpl_colors[2:]
    colormap = mpl.colors.LinearSegmentedColormap.from_list('323', colorlist, N=256, gamma=1.0)
    norm = mpl.colors.Normalize(vmax=x_cof.max(), vmin=x_cof.min())
    value2color = lambda x: colormap(norm(x))
    colors = np.array([value2color(cor) for cor in x_cof.reshape(1, -1).tolist()])\
        .reshape(x_cof.shape[0], x_cof.shape[1], 4)
    fill_colors = colors

    x_cof = np.round(x_cof, 2)

    size = x_cof
    or_size = np.nan_to_num((abs(size) / size) * (1 - abs(size)))

    n = size.shape[0]
    explode = (0, 0)
    gs = gridspec.GridSpec(n-1, n-1)
    gs.update(wspace=0.02, hspace=0.05)


    fig = plt.figure(figsize=(8, 6), frameon=True)  # args

    score_fontsize = round(12 * front_raito)
    circle_size = round(400 * front_raito)

    for i, j in product(range(0, n-1, 1), range(1, n, 1)):
        if i > j-1:
            continue
        else:
            pass

        if types == "fill":
            ax = plt.subplot(gs[i, j-1])
            ax.set_facecolor(fill_colors[i, j])

            [ax.spines[_].set_color('w') for _ in ['right', 'top', 'left', 'bottom']]

            ax.text(0.5, 0.5, size[i, j],
                    fontdict={'family': 'Arial', 'weight':'300'},  # args
                    fontsize=18,  # c_arg
                    horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])

    for k in range(0, n-1, 1):
        ax = plt.subplot(gs[k, k])
        ax.text(-0.1, 0.5, name[k], fontsize=20, horizontalalignment='right', verticalalignment='center')
    for k in range(0, n-1, 1):
        ax = plt.subplot(gs[0, k])
        ax.text(0.5, 1.2, name[k+1], fontsize=20, horizontalalignment='center', verticalalignment='center')


    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.78, 0.11, 0.04, 0.77])
    ColorbarBase(cbar_ax, cmap=colormap, norm=norm)
    ColorbarBase(cbar_ax, cmap=colormap, norm=norm)
    cf = ColorbarBase(cbar_ax, cmap=colormap, norm=norm)
    ax_cf = cf.ax
    ax_cf.tick_params(length=4, width=1.5, which='major')
    # ax_cf.set_ylim(0.46, 1.1)
    ax_cf.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_cf.set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'],
                       fontfamily='Arial', fontsize=16, fontweight='300')
    plt.text(x=2.8, y=0.78, s='Pearson coefficient',
             rotation='vertical', fontfamily='Arial',
             fontsize=20, fontweight='500', va='center')
    cf.outline.set_linewidth(1.5)
    # plt.text(x=-15.5, y=0.53, s='Corr of the conv. at different conc-0 (min)', fontfamily='Arial',
    #          fontsize=20, fontweight='500', va='center'
    #          )
    plt.show()


def correlation_plot(data):
    data =  get_slope(data)
    from scipy.stats import pearsonr
    cor_list = []
    for index_1 in range(data.shape[1]):
        for index_2 in range(data.shape[1]):
            data_1 = data[:, index_1]
            data_2 = data[:, index_2]
            cor_list.append(pearsonr(data_1, data_2)[0])
    data = np.array(cor_list).reshape(data.shape[1], data.shape[1])
    corr_plot(data, name=[2, 5, 9, 15, 22, 30], types='fill')


def distribution(data):
    mpl.rcParams["mathtext.fontset"] = 'custom'
    mpl.rcParams["mathtext.bf"] = "Arial:bold"
    mpl.rcParams["mathtext.default"] = 'regular'

    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 3.5)
    from scipy.stats import norm
    sns.distplot(data, bins=20, hist=True,
                 kde=True, ax=ax,
                 kde_kws={'linestyle': '-', 'linewidth': '0', 'color': "#4292C6"}, #'#A050A0' '#D2D2FF'
                 hist_kws={'width': 0.04, 'align': 'mid', 'color': '#BBE6FF', "edgecolor": '#000000', 'linewidth': '1.0'},
                 fit_kws={'linestyle': '-', 'linewidth': '1.5', 'color': "#4292C6"},
                 fit=norm)
    plt.hlines(y=3.0, xmin=0.45, xmax=0.5, linestyles='-', color='#4292C6', lw=2)
    plt.text(0.525, 3, s='Normal distribution',
             fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500', 'va': 'center_baseline'})
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_yticks([0, 1.0, 2.0, 3.0])
    ax.set_yticklabels([0, 1.0, 2.0, 3.0],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Density', labelpad=10, fontsize=22, fontweight='500')
    ax.set_xlabel('Conc.', labelpad=10, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')
    plt.show()


if __name__ == "__main__":
    '''figure S1'''
    # data = np.array(pd.read_csv('Fenton-data.csv'))[:, -10:]
    # conc_first = 1 - (data[:, -6]).reshape(1, -1)[0]
    # distribution(conc_first)
    # conc_all = 1 - (data[:, -6:]).reshape(1, -1)[0]
    # distribution(conc_all)

    '''figure S2-S4'''
    # data = pd.read_csv(r"/home/huatianwei/Code/Code for MEARML model/Fenton-data.csv")
    # data = np.array(data)[:, 1:]
    # condition_plot(data)
    # violin(data)
    # different_order_ort(data)
    # curves_ori = 1 - data[:, -7:]
    # cluster_plot(curves_ori, 4)




