import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def result():
    metrices = ['sensitivity', 'specificity', 'accuracy', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr']
    COO2 = np.load('pre_evaluated/OUTFF.npy', allow_pickle=True)
    stoi = [np.load('pre_evaluated/stoi__2db.npy', allow_pickle=True),
            np.load('pre_evaluated/stoi__5db.npy', allow_pickle=True)]
    pesq = [np.load('pre_evaluated/pesq_2db.npy', allow_pickle=True),
            np.load('pre_evaluated/pesq_5db.npy', allow_pickle=True)]
    ALG = ['lstm', 'cnn', 'rnn', 'bi_lstm', 'CONV_TCNN', 'PROP_TCNN']
    snr = ['-2db', '-5db']
    noise = ['AirConditioner', 'Babble', 'Munching', 'Average']
    clr, lp = ['b', 'y', 'g', 'm', 'c', 'r', '#a6e32b', '#f216ef', '#820c0c'], ['60', '70', '80', '90']
    print('MS-SNSD --Results')
    for i in range(len(metrices)):
        plt.figure()
        value1 = []
        for k in range(len(ALG)):
            value = []
            for j in range(4):
                value.append(COO2[j, k, i])
            value1.append(value)
        if i == 2:
            print('statistical analysis')
            print((pd.DataFrame(np.array([ststs(x) for x in value1]),
                                columns=['WORST', 'BEST', 'MEAN', 'MEDN', 'STND DEV'], index=ALG)).to_markdown())
        br1 = np.arange(4)
        W = 0.10
        for pt in range(len(value1)):
            br1 = np.arange(4) if pt == 0 else [x + W for x in br1]
            plt.bar(br1, value1[pt], color=clr[pt], width=W,
                    edgecolor='grey', label=ALG[pt])

        plt.subplots_adjust(bottom=0.2)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        plt.xlabel('Learning Percent (%)', fontweight='bold')
        plt.ylabel(metrices[i], fontweight='bold')
        plt.xticks([r + 0.13 for r in range(4)],
                   ['60', '70', '80', '90'])
        plt.savefig(f'results/MS-SNSD_dataset_{metrices[i]}.png')
    # for i in range(len(COO2)):
    #     lp = ['60','70','80','90']
    #     T1 = COO2[i]
    #     print(f'learning_rate/{lp[i]}')
    #     df = pd.DataFrame(COO2[i], columns=metrices,
    #                       index=ALG)
    #     print(df.to_markdown())
    Text = [[print(f'learning_rate{lp[i]}'), print(pd.DataFrame(COO2[i], columns=metrices, index=ALG).to_markdown())]
            for i
            in range(len(COO2))]
    stoii = [[print(f'STOI{snr[i]}'), print(pd.DataFrame(stoi[i], columns=noise, index=ALG).to_markdown())] for i
             in range(len(stoi))]
    pesqq = [[print(f'PESQ{snr[i]}'), print(pd.DataFrame(pesq[i], columns=noise, index=ALG).to_markdown())] for i
             in range(len(stoi))]
    plt.show(block=True)


def ststs(a):
    b = np.empty([5])
    b[0] = np.min(a)
    b[1] = np.max(a)
    b[2] = np.mean(a)
    b[3] = np.median(a)
    b[4] = np.std(a)
    return b
