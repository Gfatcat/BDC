from sklearn.datasets import fetch_20newsgroups
sample_cate = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med', 'rec.sport.baseball']
newsgroups_train = fetch_20newsgroups(subset='train',categories=sample_cate,shuffle=True, random_state=42,remove = ('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=sample_cate,shuffle=True, random_state=42,remove = ('headers', 'footers', 'quotes'))
 
print(len(newsgroups_train.data), len(newsgroups_test.data))