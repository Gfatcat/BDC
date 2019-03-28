# encoding: utf-8
from experiment import *


if __name__ == "__main__":
    train, test = loaddata("./reuters/r8-train-all-terms.txt", "./reuters/r8-test-all-terms.txt")

    package = ()
    package = createdic(train, test)
    schemes = ["tf", "tf_idf", "tf_ig", "tf_chi", "tf_rf", "iqf_qf_icf", "tf_eccd", "dc", "bdc", "tf_dc", "tf_bdc"]


    import time
    now = str(time.strftime('_%m%d_%H%M',time.localtime(time.time())))
    path = ("./results"+now+"/")

    if not os.path.exists(path):
        os.mkdir(path)

    import multiprocessing
    pool = multiprocessing.Pool()
    for sche in schemes:
        pool.apply_async(experiment, args=(package, sche, "all", path))

    pool.close()
    pool.join()
