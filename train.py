import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import time

def train(model_name,test_s):
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H.%M.%S", t)
    name = model_name + "_" + str(current_time) + ".pkl"
    if model_name == "HOG":
        df = pd.read_csv('CSV/dataset_HOG.csv')
        c = 1.0
    elif model_name == "LBP":
        df = pd.read_csv('CSV/dataset_LBP.csv')
        c = 10.0    
    df.columns = [i for i in range(df.shape[1])]
    df = df.rename(columns={df.columns[-1]: 'Etiket'})

    X = df.iloc[:, :-1]
    print("Features shape =", X.shape)

    Y = df.iloc[:, -1]
    print("Labels shape =", Y.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_s, random_state=1)

    svm = SVC(C=c, gamma=0.1, kernel='linear')
    svm.fit(x_train, y_train)

    y_pred = svm.predict(x_test)

    cf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    print("Accuracy score %.3f" %accuracy_score(y_test, y_pred))
    print("f1, recall, precision %.3f" %f1,recall,precision)

    labels = sorted(list(set(df['Etiket'])))
    labels = [x.upper() for x in labels]

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.set_title("Confusion Matrix")

    maping = sns.heatmap(cf_matrix, 
                        annot=True,
                        cmap = plt.cm.Blues, 
                        linewidths=.2,
                        xticklabels=labels,
                        yticklabels=labels, vmax=8,
                        fmt='g',
                        ax=ax
                        )
    plt.savefig("confusion.png")
    # maping  #GORSELLESTIRME

    import pickle

    # save model
    with open("MODEL/"+name,'wb') as f:
        pickle.dump(svm,f)
    print("Egitilmis Model Kaydedildi. Bknz:MODEL/"+name)
    return maping,accuracy_score(y_test, y_pred),recall,precision