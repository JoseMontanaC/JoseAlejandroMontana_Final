import numpy as np
import matplotlib.pylab as plt
cadenas=[]
degrade= 0.2
color=["plum","m","darkgreen","firebrick","salmon","magenta","peru","c"]
aux=0
for i in range(1,9):
    archivo=np.genfromtxt("cadena_"+str(i)+".txt")
    plt.hist(archivo,bins=50,density=True,alpha=degrade)
    cadenas.append(archivo)
    # degrade +=0.1
    aux+=1
plt.show()


