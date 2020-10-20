##
import functools
my_list=[1,4,5,2,3]
def add_it(x,y):
    return(x+y)
sum = functools.reduce(add_it, my_list)
print(sum)

##
sum = 0
for x in my_list:
    sum += x
print(sum)

##
print (do_add([1,10,100,1000]))

##
arr=[1,2,3,4,5]
for i in arr:
    print(i," ",end="-->")

##
#print a powerfull function
day="monday"
print("today is {} and we have {} degrees. It was cold")

##
print(type(arr))

##
color_list=["Red","Green","White","Black"]
def first_and_last(any_list):
    for x in color_list:
        print(color_list[0], color_list[len(color_list)-1])
first_and_last(color_list)

##
x=input()
y=input()
print(y," ",x)

##
x=int(input())
if x<27:
    y=27-x
else:
    y=2*(x-27)
print(y)

##
x=int(input())
if x%2 == 0:
    print("even")
else:
    print("odd")

##
a=[1,2,4,4]
len(a)

##
import numpy as np
print(np.__version__)
a=np.array([1,2,4,4])
type(a)
a
print(a.size)
print(a.shape)

##
print(len(a))
print(np.count_nonzero(a==4,axis=0))
a[3]=5
print(a)

##
a.shape=(2,2)
print(a)
print(a.T)
print(np.linalg.det(a))
print(np.linalg.det(a.T))

##
a.shape=(1,4)
print(a)

##
a.T.shape
a

##
print(np.sort(a))

##
print(format(a.mean()))
print(a.std())

##
boolarr=a>1
print(boolarr)
#crée une matrice avec vrai faux
print(a[(a<=4)&(a>=2)])
#supprime certaines valeurs en fonction de ce qu'on demande

##
index=np.where(a>1)
list(index[0])

##
a[a>2]*=3
print(a)
#toutes les valeurs superieures à 2 sont toutes multipliées par 3

##
ran=np.random.randint(0,10,100,int)
print(ran)
#crée une matrice random

##
vec=np.arange(15,56,1)
print(vec)

##
print(vec[4:-5])
#retire les 5 dernieres valeurs et les 4 premieres

##
np.eye(4)
#matrice identité

##
ran=np.random.randint(0,5,(3,3),int)
print(ran)
#crée une matrice random de taille 3,3 av entiers compris entre 0 et 5

##
import numpy as np
arr1=np.array([1,3,4,5,6])
arr2=np.array([2,1,0,0,2])
arr3=np.multiply(arr1,arr2)
print(arr3)
#multiplication de matrices

##
mat=np.array([[1,2],[3,5]])
print(mat)
print(mat.T)
#transposée
print(np.linalg.inv(mat))

##
mat3=np.dot(mat,np.linalg.inv(mat))
print(mat3)

#PANDAS

##
import pandas as pd
print(pd.__version__)

##
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
print(df.head())
print(df)
#affichage de la dataframe

##
print (df.keys())

##
n=["Sepal Length","Sepal Width","Petal Length","Petal Width","Class"]
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=n)
print(df.head())

##
print("Petal Length Mean:",df["Petal Length"].mean())
print("Sepal Length Mean:",df["Sepal Length"].mean())
print("Petal Width Mean:",df["Petal Width"].mean())
print("Sepal Width Mean:",df["Sepal Width"].mean())

##
#Convert a Class object to a numeric
cl=pd.Categorical(df["Class"])
print(cl.codes)
df["Class Code"]=cl.codes
print(df)

##
#Correlate
print(df["Petal Length")
         # pas fini

##
d= {'col1':[1,2],'col2':[3,4]}

print(d)

#MATPLOTLIB

##
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(0,10,100)
print(x)
y=plt.plot(x,3+2*x,"b-")
#(x=axe des abscisses/3+2*x=axe des ordonnés/r=couleur red/-=ligne)
y

##
y=plt.plot(x,3+2*x,"b-",x,-2+3*x,"r-")
y

##
y=plt.plot(x,np.cos(x),"b-")
y

##
#Ex1
a = str("ArtificialIntelligence")
for i = 0 to 10
a = a[2 * i]

##
#Ex3
x=int(input())
y=int(input())
if x<=y:
    print("Max =", y)
else:
    print("Max =", x)

##
#Ex4
x = input()
def f(x):
    for i in range(len(x) // 2):
        if x[i] != x[-i - 1]:
            return False
    return True
f(x)