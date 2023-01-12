import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import copy


def main():

    # visual_lrg_mul_image()

    # visual_single_image()

    # trans_list = ['AWRG', 'RG', 'GC', 'GT', 'FC']
    # for trans in trans_list:
    #     concat(trans)

    plot_confusion_matrix()

    # report()

    # report_mode()

    # plot_current()

    # plt_color_map()

    # confusion_convert_pre_and_trg()

    # plt_bar()

    # plt_train_val()

    # test_game()


def test_game():
    pre = np.load('./pre.npy')
    trg = np.load('./true.npy')

    cm = confusion_matrix(trg, pre)
    cmp = plt.cm.Blues
    # ax = plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    plt.xticks([])
    plt.yticks([])

    cm_max = np.max(cm)
    cm_min = np.min(cm)
    cm_hat = copy.deepcopy(cm)
    cmp = [cmp(i) for i in np.linspace(0.1, 1, cm_max-cm_min+1)]
    image = np.zeros(shape=(12, 12, 3))
    for i in range(len(image)):
        for j in range(len(image)):
            if 1 <= cm[i, j] <= 5:
                cm_hat[i, j] += 10

            image[i, j, :] = cmp[cm_hat[i, j]][:-1]
    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            if cm[first_index][second_index] > 30:
                color = 'w'
            else:
                color = None
            plt.text(second_index, first_index, int(cm[first_index][second_index]),
                     horizontalalignment='center', verticalalignment='center', color=color)
    plt.imshow(image)
    plt.show()
    rf_report = classification_report(trg, pre, digits=10, output_dict=True)
    rf_score_list = [v['f1-score']-0.90 for k, v in rf_report.items() if len(k) < 5][::-1]
    plt.figure(figsize=[8, 10])
    y = [x*2+3 for x in range(12)]
    plt.barh(y=y, width=rf_score_list, align="center", color="slategray", alpha=1)
    plt.yticks([])
    plt.xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1], ['$90$', '$92$', '$94$', '$96$', '$98$', '$100$'], fontsize=15)
    plt.show()
    # print(report)


def plt_train_val():
    rg_path = './train_val_loss/rg_cnn.pickle'
    gru_path = './train_val_loss/rg_GRU.pickle'
    lstm_path = './train_val_loss/rg_LSTM.pickle'
    att_path = './train_val_loss/rg_Transformer.pickle'
    with open(rg_path, 'rb') as handle:
        rg_train_acc, rg_val_acc, rg_train_loss, rg_val_loss = pickle.load(handle)
        rg_train_acc = np.array(rg_train_acc)
        rg_val_acc = np.array(rg_val_acc)
        rg_train_loss = np.array(rg_train_loss)
        rg_val_loss = np.array(rg_val_loss)
    with open(gru_path, 'rb') as handle:
        gru_train_acc, gru_val_acc, gru_train_loss, gru_val_loss = pickle.load(handle)
        gru_train_acc = np.array(gru_train_acc)
        gru_val_acc = np.array(gru_val_acc)
        gru_train_loss = np.array(gru_train_loss)
        gru_val_loss = np.array(gru_val_loss)
    with open(lstm_path, 'rb') as handle:
        lstm_train_acc, lstm_val_acc, lstm_train_loss, lstm_val_loss = pickle.load(handle)
        lstm_train_acc = np.array(lstm_train_acc)
        lstm_val_acc = np.array(lstm_val_acc)
        lstm_train_loss = np.array(lstm_train_loss)
        lstm_val_loss = np.array(lstm_val_loss)
    with open(att_path, 'rb') as handle:
        att_train_acc, att_val_acc, att_train_loss, att_val_loss = pickle.load(handle)
        att_train_acc = np.array(att_train_acc)
        att_val_acc = np.array(att_val_acc)
        att_train_loss = np.array(att_train_loss)
        att_val_loss = np.array(att_val_loss)
    x = [x for x in range(1000)][::4]
    plt.subplot(2, 2, 1)
    plt.plot(x, rg_train_acc[::4] * 100, color='r')
    plt.plot(x, rg_val_acc[::4] * 100, color='b')

    plt.subplot(2, 2, 2)
    plt.plot(x, gru_train_acc[::4] * 100, color='r')
    plt.plot(x, gru_val_acc[::4] * 100, color='b')

    plt.subplot(2, 2, 3)
    plt.plot(x, lstm_train_acc[::4] * 100, color='r')
    plt.plot(x, lstm_val_acc[::4] * 100, color='b')

    plt.subplot(2, 2, 4)
    plt.plot(x, att_train_acc[::4] * 100, color='r')
    plt.plot(x, att_val_acc[::4] * 100, color='b')
    plt.show()


def plt_bar():
    rg_list = [98.63, 87.72, 96.94, 95.57]
    gc_list = [98.57, 90.22, 92.89, 92.50]
    gt_list = [98.59, 88.30, 97.79, 95.83]
    fc_list = [98.37, 96.09, 96.53, 97.02]
    total = np.array([rg_list, gc_list, gt_list, fc_list]) - 80
    total = total.T
    plt.figure(figsize=(8, 5))
    x = np.arange(4)
    total_width, n = 0.6, 4
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.bar(x, total[0], color='crimson', width=width-0.03, label='TCN')  # teal
    plt.bar(x + width, total[1], color='k', width=width-0.03, label='LSTM')  # y
    plt.bar(x + 2*width, total[2], color='b', width=width-0.03, label='GRU')  # c
    plt.bar(x + 3*width, total[3], color='blueviolet', width=width-0.03, label='ATT')  # coral
    plt.xticks([0, 1, 2, 3], ['LRG', 'LGC', 'LGT', "GG"], fontsize=15)
    plt.yticks([0, 5, 10, 15, 20], ['80', '85', '90', '95', '100'], fontsize=15)
    plt.legend(loc=(0.42, 0.78))
    plt.tight_layout()

    plt.show()


def report():
    path_awrg = './pre_and_trg/plaid2018_pre_and_trg_AWRG.pickle'
    path_rg = './pre_and_trg/plaid2018_pre_and_trg_RG.pickle'
    path_gc = './pre_and_trg/plaid2018_pre_and_trg_GC.pickle'
    path_gt = './pre_and_trg/plaid2018_pre_and_trg_GT.pickle'
    path_fc = './pre_and_trg/plaid2018_pre_and_trg_FC.pickle'

    with open(path_awrg, 'rb') as handle:
        pre, trg = pickle.load(handle)
    awrg_report = classification_report(trg, pre, digits=10, output_dict=True)
    with open(path_rg, 'rb') as handle:
        pre, trg = pickle.load(handle)
    rg_report = classification_report(trg, pre, digits=10, output_dict=True)
    with open(path_gc, 'rb') as handle:
        pre, trg = pickle.load(handle)
    gc_report = classification_report(trg, pre, digits=10, output_dict=True)
    with open(path_gt, 'rb') as handle:
        pre, trg = pickle.load(handle)
    gt_report = classification_report(trg, pre, digits=10, output_dict=True)
    with open(path_fc, 'rb') as handle:
        pre, trg = pickle.load(handle)
    fc_report = classification_report(trg, pre, digits=10, output_dict=True)

    awrg_score_list = [v['f1-score']-0.91 for k, v in awrg_report.items() if len(k) < 5][::-1]
    rg_score_list = [v['f1-score']-0.91 for k, v in rg_report.items() if len(k) < 5][::-1]
    gc_score_list = [v['f1-score']-0.91 for k, v in gc_report.items() if len(k) < 5][::-1]
    gt_score_list = [v['f1-score']-0.91 for k, v in gt_report.items() if len(k) < 5][::-1]
    fc_score_list = [v['f1-score']-0.91 for k, v in fc_report.items() if len(k) < 5][::-1]

    plt.figure(figsize=[8, 10])
    y = [x*5+3 for x in range(11)]
    plt.barh(y=y, width=awrg_score_list, align="center", color="k", alpha=1)

    y = [x*5+2 for x in range(11)]
    plt.barh(y=y, width=rg_score_list, align="center", color="r", alpha=1)
    # crimson
    y = [x*5+1 for x in range(11)]
    plt.barh(y=y, width=gc_score_list, align="center", color="crimson", alpha=1)
    # c
    y = [x*5 for x in range(11)]
    plt.barh(y=y, width=gt_score_list, align="center", color="firebrick", alpha=1)
    # orange
    # y = [x*5 for x in range(11)]
    # plt.barh(y=y, width=fc_score_list, align="center", color="r", alpha=1)
    # slategray
    plt.plot([0.0768, 0.0768], [-1, 53.5], color='crimson', linewidth=1)

    # ax = plt.gca()
    # r_rg = plt.Rectangle((0.08, 17), 0.005, 0.5, color='crimson', linewidth=1)
    # r_gc = plt.Rectangle((0.08, 15), 0.005, 0.5, color='c', linewidth=1)
    # r_gt = plt.Rectangle((0.08, 13), 0.005, 0.5, color='orange', linewidth=1)
    # r_fc = plt.Rectangle((0.08, 11), 0.005, 0.5, color='slategray', linewidth=1)
    # ax.add_patch(r_rg)
    # ax.add_patch(r_gc)
    # ax.add_patch(r_gt)
    # ax.add_patch(r_fc)

    plt.yticks([])
    plt.xticks([0, 0.03, 0.06, 0.09], ['$91$', '$94$', '$97$', '$100$'], fontsize=20)
    plt.show()


def report_mode():
    path_rg = './pre_and_trg/plaid2018_pre_and_trg_rg.pickle'
    path_gc = './pre_and_trg/plaid2018_pre_and_trg_lstm.pickle'
    path_gt = './pre_and_trg/plaid2018_pre_and_trg_gru.pickle'
    path_fc = './pre_and_trg/plaid2018_pre_and_trg_transformer.pickle'

    with open(path_rg, 'rb') as handle:
        pre, trg = pickle.load(handle)
    rg_report = classification_report(trg, pre, digits=10, output_dict=True)
    with open(path_gc, 'rb') as handle:
        pre, trg = pickle.load(handle)
    gc_report = classification_report(trg, pre, digits=10, output_dict=True)
    with open(path_gt, 'rb') as handle:
        pre, trg = pickle.load(handle)
    gt_report = classification_report(trg, pre, digits=10, output_dict=True)
    with open(path_fc, 'rb') as handle:
        pre, trg = pickle.load(handle)
    fc_report = classification_report(trg, pre, digits=10, output_dict=True)

    rg_score_list = [v['f1-score'] for k, v in rg_report.items() if len(k) < 5]
    gc_score_list = [v['f1-score'] for k, v in gc_report.items() if len(k) < 5]
    gt_score_list = [v['f1-score'] for k, v in gt_report.items() if len(k) < 5]
    fc_score_list = [v['f1-score'] for k, v in fc_report.items() if len(k) < 5]

    plt.figure(figsize=[8, 10])
    y = [x*5+3 for x in range(11)]
    plt.barh(y=y, width=np.array(rg_score_list)-0.85, align="center", color="crimson", alpha=1)

    y = [x*5+2 for x in range(11)]
    plt.barh(y=y, width=np.array(gc_score_list)-0.85, align="center", color="c", alpha=1)

    y = [x*5+1 for x in range(11)]
    plt.barh(y=y, width=np.array(gt_score_list)-0.85, align="center", color="orange", alpha=1)

    y = [x*5 for x in range(11)]
    plt.barh(y=y, width=np.array(fc_score_list)-0.85, align="center", color="slategray", alpha=1)

    plt.plot([0.1315, 0.1315], [-1, 53.5], color='crimson', linewidth=1)

    ax = plt.gca()
    r_rg = plt.Rectangle((0.08, 17), 0.005, 0.5, color='crimson', linewidth=1)
    r_gc = plt.Rectangle((0.08, 15), 0.005, 0.5, color='c', linewidth=1)
    r_gt = plt.Rectangle((0.08, 13), 0.005, 0.5, color='orange', linewidth=1)
    r_fc = plt.Rectangle((0.08, 11), 0.005, 0.5, color='slategray', linewidth=1)
    ax.add_patch(r_rg)
    ax.add_patch(r_gc)
    ax.add_patch(r_gt)
    ax.add_patch(r_fc)

    plt.yticks([])
    plt.xticks([0, 0.05, 0.10, 0.15], ['$85$', '$90$', '$95$', '$100$'], fontsize=15)
    plt.show()


def plt_color_map():
    x = np.linspace(0, 1, 10)
    number = 5
    cmp = plt.get_cmap('gnuplot')

    colors = [cmp(i) for i in np.linspace(0, 1, number)]

    for i, color in enumerate(colors, start=1):
        plt.plot(x, i * x + i, color=color, label='$y = {i}x + {i}$'.format(i=i))
    plt.legend(loc='best')
    plt.show()


def visual_single_image():
    path_image = './image_out/plaid2018_fc.pickle'
    # path_label = './image_out/plaid2018_rg_label.pickle'

    with open(path_image, 'rb') as handle:
        lrg_image = pickle.load(handle)

    # with open(path_label, 'rb') as handle:
        # lrg_label = pickle.load(handle)

    lrg_image = lrg_image.to('cpu')
    lrg_image = np.array(lrg_image)

    plt.figure(figsize=[10, 10])
    plt.imshow(lrg_image[1][0], cmap='pink_r')
    ax = plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    plt.xticks([0, 10, 20], fontsize=30)
    plt.yticks([0, 10, 20], fontsize=30)
    plt.show()


def plot_current():

    current = np.load('./data/plaid2018/aggregated/current.npy')

    fig, ax = plt.subplots()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.plot(current[1], color='r', linewidth=1)
    plt.show()


def plot_confusion_matrix():
    path_rg = './pre_and_trg/plaid2018_pre_and_trg_RG.pickle'
    path_gc = './pre_and_trg/plaid2018_pre_and_trg_GC.pickle'
    path_gt = './pre_and_trg/plaid2018_pre_and_trg_GT.pickle'
    path_fc = './pre_and_trg/plaid2018_pre_and_trg_FC.pickle'
    cm_color_rg, cm_rg = generate_color_confusion(path_rg, cmp=plt.cm.gray_r)
    cm_color_gc, cm_gc = generate_color_confusion(path_gc, cmp=plt.cm.Greens)
    cm_color_gt, cm_gt = generate_color_confusion(path_gt, cmp=plt.cm.Reds)
    cm_color_fc, cm_fc = generate_color_confusion(path_fc, cmp=plt.cm.Blues)
    cm_color = generate_all_color_confusion(rg=cm_color_rg, gc=cm_color_gc, gt=cm_color_gt, fc=cm_color_fc)
    cm_all = generate_all_confusion(rg=cm_rg, gc=cm_gc, gt=cm_gt, fc=cm_fc)
    plt.imshow(cm_color)

    for first_index in range(len(cm_all)):  # 第几行
        for second_index in range(len(cm_all[first_index])):  # 第几列
            if cm_all[first_index][second_index] > 100:
                color = 'w'
            else:
                color = None
            plt.text(second_index, first_index, int(cm_all[first_index][second_index]),
                     horizontalalignment='center', verticalalignment='center', color=color)
    for i in range(0, 10):
        plt.plot([-0.5, 21.5], [1.45 + (i * 2), 1.45 + (i * 2)], color='black', linewidth=1)
        plt.plot([1.5 + (i * 2), 1.5 + (i * 2)], [-0.5, 21.5], color='black', linewidth=1)
    plt.show()


def generate_all_confusion(rg, gc, gt, fc):
    cm_all = np.zeros(shape=(22, 22))
    for i in range(0, len(cm_all), 2):
        for j in range(0, len(cm_all), 2):
            cm_all[i, j] = rg[i//2, j//2]

    for i in range(1, len(cm_all), 2):
        for j in range(0, len(cm_all), 2):
            cm_all[i, j] = gc[i//2, j//2]

    for i in range(0, len(cm_all), 2):
        for j in range(1, len(cm_all), 2):
            cm_all[i, j] = gt[i//2, j//2]

    for i in range(1, len(cm_all), 2):
        for j in range(1, len(cm_all), 2):
            cm_all[i, j] = fc[i//2, j//2]

    return cm_all


def generate_all_color_confusion(rg, gc, gt, fc):
    cm_all = np.zeros(shape=(22, 22, 3))
    for i in range(0, len(cm_all), 2):
        for j in range(0, len(cm_all), 2):
            cm_all[i, j, :] = rg[i//2, j//2, :]

    for i in range(1, len(cm_all), 2):
        for j in range(0, len(cm_all), 2):
            cm_all[i, j, :] = gc[i//2, j//2, :]

    for i in range(0, len(cm_all), 2):
        for j in range(1, len(cm_all), 2):
            cm_all[i, j, :] = gt[i//2, j//2, :]

    for i in range(1, len(cm_all), 2):
        for j in range(1, len(cm_all), 2):
            cm_all[i, j, :] = fc[i//2, j//2, :]

    return cm_all


def generate_color_confusion(path=None, cmp=None):

    with open(path, 'rb') as handle:
        pre, trg = pickle.load(handle)

    cm = confusion_matrix(trg, pre)
    # ax = plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    plt.xticks([])
    plt.yticks([])

    cm_max = np.max(cm)
    cm_min = np.min(cm)
    cm_hat = copy.deepcopy(cm)
    cmp = [cmp(i) for i in np.linspace(0.1, 1, cm_max-cm_min+1)]
    image = np.zeros(shape=(11, 11, 3))
    for i in range(len(image)):
        for j in range(len(image)):
            if 1 <= cm[i, j] <= 10:
                cm_hat[i, j] += 30

            image[i, j, :] = cmp[cm_hat[i, j]][:-1]

    return image, cm


def concat(trans):
    prediction = np.zeros(shape=(1824, 1))
    target = np.zeros(shape=(1824, 1))

    for i in range(10):
        with open(f'./plaid2018_pre_and_trg_i/plaid2018_pre_and_trg{trans}_{i}.pickle', 'rb') as handle:
            pre, trg = pickle.load(handle)
        pre = np.array(pre)
        trg = np.array(trg)
        start = i * len(pre)
        end = (i+1) * len(pre)
        prediction[start: end, :] = pre[:, :]
        target[start: end, :] = trg[:, :]
    with open(f'./plaid2018_pre_and_trg_{trans}.pickle', 'wb') as handle:
        pickle.dump((prediction, target), handle, protocol=2)
    f_macro = f1_score(target, prediction, average='macro')
    print(f_macro)


def visual_lrg_mul_image():

    path_image = './image_out/plaid2018_fc.pickle'
    path_label = './image_out/plaid2018_fc_label.pickle'

    with open(path_image, 'rb') as handle:
        lrg_image = pickle.load(handle)

    with open(path_label, 'rb') as handle:
        lrg_label = pickle.load(handle)

    app_list = [i for i, app_id in enumerate(lrg_label) if app_id == 11][:20]
    lrg_image = lrg_image.to('cpu')
    lrg_image = np.array(lrg_image)
    app_image = lrg_image[app_list, :, :, :]
    plt.figure(figsize=[10, 10])
    for i in range(len(app_image)):

        plt.subplot(4, 5, i+1)
        plt.imshow(app_image[i][0], cmap='Blues')
        plt.colorbar(orientation='horizontal')
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        plt.xticks([])
        plt.yticks([])
    plt.show()


def confusion_convert_pre_and_trg():
    cm = np.array([[62, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                   [0, 174, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 110, 0, 1, 0, 4, 0, 0, 0, 0],
                   [7, 0, 3, 23, 1, 0, 3, 0, 0, 1, 0],
                   [0, 0, 0, 0, 153, 3, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 114, 0, 0, 0, 0],
                   [0, 2, 0, 0, 0, 1, 0, 169, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 139, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0],
                   [2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 23]])

    cm1 = np.array([[63, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                   [0, 174, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 115, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2, 0, 2, 30, 1, 0, 2, 0, 1, 0, 0],
                   [0, 0, 0, 0, 153, 3, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 114, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 171, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 139, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0],
                   [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23]])

    trg = np.zeros(shape=np.sum(cm1))
    pre = np.zeros(shape=np.sum(cm1))
    count = 0
    for i, r in enumerate(cm1):
        for j, c in enumerate(r):
            pre[count:count+c] = [j for _ in range(c)]
            count += c
    count = 0
    for i, r in enumerate(cm1):
        trg[count:count+np.sum(r)] = [i for _ in range(np.sum(r))]
        count += np.sum(r)

    result = classification_report(trg, pre, digits=10)
    cm_1 = confusion_matrix(trg, pre)
    print(cm_1)
    print(result)


if __name__ == '__main__':
    main()
