import numpy as np
import matplotlib.pyplot as plt

def draw(result_dir):
    # 绘制knn结果
    schemes = ["tf", "tf_idf", "tf_chi", "tf_ig", "tf_eccd", "tf_rf", "iqf_qf_icf", "dc", "bdc", "tf_dc", "tf_bdc"]
    color = ['blue', 'green', 'red', 'cyan', 'purple', 'gold', 'grey', 'blue', 'green', 'red', 'cyan']
    mark = ['o', 's', 'x', '.', '+', '*', 'D', 'v', '^', '<', '>']
    plt.figure()
    # ax.bar(x, befor,  width=width, label='befor',color='b')
    for i in range(len(schemes)):
        resultlist = [w.strip().split('\t') for w in open(result_dir + "knn" + "_" + schemes[i] + ".txt",
                'r', encoding='utf-8').readlines()]
        resultarray = np.array(resultlist,dtype=float)
        plt.plot(resultarray[:, 0], resultarray[:, 1],color= color[i],marker=mark[i], label=schemes[i])
        # j =  range(resultarray.shape[0])
        # plt.scatter(resultarray[j, 0], resultarray[j, 1],color='', edgecolors=color[i], marker=mark[i], label=schemes[i])
#     plt.ylim((0.79, 1.0))
    x_ticks = [i for i in range(0, 76, 5)]
    x_ticks[0] = 1
    plt.xlabel("K")
    plt.ylabel("MicrpF1")
#     y_ticks = np.arange(0.79,1.01,0.05)
#     plt.yticks = (y_ticks)
    plt.xticks = (x_ticks)
    plt.legend(ncol=11,loc ="lower center")
    plt.show()
    plt.savefig(result_dir+"fig.png")

def svm_result(result_dir):
    schemes = ["tf", "tf_idf", "tf_chi", "tf_ig", "tf_eccd", "tf_rf", "iqf_qf_icf", "dc", "bdc", "tf_dc", "tf_bdc"]
    resultfile = open(result_dir + "svm_result.txt", "w")
    for i in range(len(schemes)):
        resultlist = [w.strip().split('\t') for w in open(result_dir + "svm" + "_" + schemes[i] + ".txt",'r', encoding='utf-8').readlines()]
        resultarray = np.array(resultlist,dtype=float)
        micro_f1 = resultarray[0,0]
        macro_f1 = resultarray[0,1]
        resultfile.write(schemes[i]+"\t"+str(micro_f1) + "\t" + str(macro_f1) + "\n")
    resultfile.close()
        

if __name__ == "__main__":
        diretion = "./results_0326_1320/"
        draw(diretion)
        svm_result(diretion)