import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score ,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
from matplotlib.colors import ListedColormap

def Classification(X_train, X_test, Y_train, Y_test):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    print("The Accuracy Is:",accuracy_score(Y_test, Y_pred))
    return accuracy_score

    

def Naive_Bayes(X_train, X_test, Y_train, Y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    print(f'The output of Naive Bayes:\n{cm}')
    return cm

def aiFunctoin():
    y= []
    for i in y:
        i = ''
    csv_file = "c:/Users/Amit/OneDrive/desktop/Kinneret/ai/AIfinal/datasheetwoman.csv"
    dataset = pd.read_csv(csv_file)
    
    dataset.Year = [datetime.strptime(year, '%d/%m/%Y').timestamp() for year in dataset.Year]
    dataset = dataset.replace('unknown', np.NaN)
    
    Region = {'south' : 1, 'north' : 2, 'central' : 3, 'global' : 4, 'ramle' : 5, np.NaN : np.NaN}
    dataset.Region = [Region[item] for item in dataset.Region]
    y.append(dataset.iloc[:, 3].values)

    Origin = {'IL' : 1, 'ILArab' : 2, 'russian' : 3, 'stranger' : 4, 'nigger' : 5, np.NaN : np.NaN}
    dataset.Origin = [Origin[item] for item in dataset.Origin]
    y.append(dataset.iloc[:, 4].values)

    Relagion = {'jew' : 1, 'muslim' : 2 , 'badui' : 3, 'druz' : 4, 'arith' : 5, 'Philip' : 6 ,np.NaN : np.NaN}
    dataset.Relagion = [Relagion[item] for item in dataset.Relagion]
    y.append(dataset.iloc[:, 5].values)

    Killer = {'husband' : 1, 'devorce' : 2, 'partner' : 3,'son' : 4, 'brother' : 5, 'son husband' : 6,'father brother neice' : 7, 'brother father' : 8, 'relatives' : 9,'father' : 10,'dautgher partner' : 11, 'causin' : 12,'neighbor' : 13,'devorce causin' : 14,'motherinlaw' : 15,'fionse' : 16,'hitman' : 17,'expartner' : 18, 'devorce brodevorce causin husband' : 19, 'dautgherofex' : 20 , 'bro threefriends' : 21, 'student' : 22, 'daugther' : 23, 'Sister Husband' : 24, 'mother' : 25, np.NaN : np.NaN}
    dataset.Killer = [Killer[item] for item in dataset.Killer]
    y.append(dataset.iloc[:, 7].values)

    Weapon = {'bruteforce' : 1 ,'sharptool' : 2 , 'knife' : 3 , 'pistol' : 4 , 'hammer' : 5 , 'byhands' : 6 , 'gun' : 7 , 'burnalive' : 8 , np.NaN : np.NaN}
    dataset.Weapon = [Weapon[item] for item in dataset.Weapon]
    y.append(dataset.iloc[:, 8].values)

    Known = {'no' : 1 , 'yes' : 2 , np.NaN : np.NaN}
    dataset.Known = [Known[item] for item in dataset.Known]
    y.append(dataset.iloc[:, 9].values)

    IsWoman = {'no' : 1 , 'yes' : 2 , np.NaN : np.NaN}
    dataset.IsWoman = [IsWoman[item] for item in dataset.IsWoman]
    y.append(dataset.iloc[:, 10].values)

    Status = {'suspect arrested' : 1 , 'arrested' : 2 , 'uncaught' : 3 , 'kill him self' : 4 , 'prison' : 5 , np.NaN : np.NaN}
    dataset.Status = [Status[item] for item in dataset.Status]
    y.append(dataset.iloc[:, 11].values)

    WhereBody = {'home' : 1 , 'street' : 2 , 'stairs' : 3 , 'hospital' : 4 , 'car' : 5 , 'office' : 6 , 'graveyard' : 7 , 'hotel' : 8 , 'forest' : 9 , 'garden' : 10 , 'factory' : 11 , 'busstation' : 12 , 'taxi' : 13 , 'hole' : 14 , 'road' : 15 , np.NaN : np.NaN}
    dataset.WhereBody = [WhereBody[item] for item in dataset.WhereBody]
    y.append(dataset.iloc[:, 12].values)

    Background = {'broken heart' : 1, 'brother threat' : 2, 'castidy' : 3, 'conflict' : 4, 'fight' : 5, 'marrige fight' : 6, 'mother son conflict' : 7, 'partnerconflict' : 8, 'racism' : 9, 'relatives fight' : 10, 'relatives fight ' : 11, 'two brother attacked mother' : 11, np.NaN : np.NaN}
    dataset.Background = [Background[item] for item in dataset.Background]

    BodyTime = {'hours': 1,'days' : 2, 'months' : 3, 'weeks' : 4, np.NaN : np.NaN}
    dataset.BodyTime = [BodyTime[item] for item in dataset.BodyTime]
    y.append(dataset.iloc[:, 14].values)

    dataset.fillna(dataset.median(), inplace=True)

    output = pd.DataFrame(dataset)
    output.to_csv("c:/Users/Amit/OneDrive/desktop/Kinneret/ai/AIfinal/output_datasheetwoman.csv", index=False)

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    for i in y:
        print('\n|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|\n')
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, i, test_size = 0.2, random_state = 0)
        aa = Classification(X_train, X_test, Y_train, Y_test)
        bb = Naive_Bayes(X_train, X_test, Y_train, Y_test)

    return dataset






dataset = aiFunctoin()