import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

fs = 20

def ej1():
    A = np.array([[1,0,1],[2,-1,1],[-3,2,-2]])
    b = np.array([-2,1,-1])
    v = np.linalg.solve(A, b)
    print( "La solucion del ejercicio 1 es {}".format(v) )

def ej2():

    data = np.random.gamma(3,2,100)
    media  = round(np.mean(data),2)
    sigma = round(np.std(data),2)
    print("Media calculada = {} --  Sigma calculada = {}".format(media,sigma))

    plt.figure(figsize=(6,6))
    sns.set(style='whitegrid')
    plt.hist(data)
    plt.xticks(np.arange(0,21,5),fontsize=fs)
    plt.yticks(fontsize=fs)
    ymax = plt.ylim()[1]
    xmax = plt.xlim()[1]
    plt.vlines(media,ymin=0,ymax=ymax,linestyle='--')
    plt.title(r"$\mu$ = {} $\sigma$ = {}".format(media,sigma),fontsize=fs)
    plt.ylim(0,ymax)
    #plt.savefig('ej1_2.pdf')
    plt.show()

def ej3(a,b=0,c=0):
    if a == 0 or b**2-4*a*c < 0:
        print("Polinomio invalido o sin raices reales")
        return []
    x1=(-b-np.sqrt(b**2-4*a*c))/(2*a)
    x2=(-b+np.sqrt(b**2-4*a*c))/(2*a)
    x = [x1,x2]
    return [min(x),max(x)]

def ej4(a,b=0,c=0):
    
    def pol(a,x,b=0,c=0):
        return a*x**2+b*x+c

    def plot_pol (a, b=0, c=0):

        plt.figure(figsize=(10, 8))
        sns.set(style='whitegrid')
        ax = plt.subplot(111)
        x = np.linspace(-4.5,4,100)
        y = pol(a,x,b,c)
        plt.plot(x,y,c='red',lw='4',label='Polinomio')
        ax.tick_params(axis='both', labelsize=15)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('data',0))
        ax.spines['left'].set_position(('data',0))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params('both',labelsize=fs)
        

        roots = ej3(a,b,c)

        if len(roots)==0: ##si el polinomio es invalido 
            plt.xlim(-4.5,4)
            plt.show()
            return 0
        
        x1 , x2 = roots[0], roots[1]
        plt.scatter([x1,x2],[pol(a,x1,b,c),pol(a,x1,b,c)],100,c='red')

        s=[r'$x_{} = {}$'.format(i,roots[i-1].round(3)) for i in [1,2]]

        if (x1 > -4.5) and (x1 < 4):
            plt.annotate(s[0],xy=(x1,pol(a,x1,b,c)),xycoords='data',xytext=(0.275,0.5),textcoords='figure fraction', fontsize=fs,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        if (x2 > -4.5) & (x2 < 4): 
            plt.annotate(s[1],xy=(x2,pol(a,x2,b,c)),xycoords='data',xytext=(0.6,0.5),textcoords='figure fraction', fontsize=fs,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        plt.xlim(-4.5,4)
        #plt.savefig('ej4_2.pdf')
        plt.show()
        
    
    plot_pol(a,b,c)

def ej5_6(a,b,x): ##pasar los valores de a,b y donde se quieren evaluar
    class Lineal:
        def __init__(self,a,b):
            self.a = a  
            self.b = b

        def __call__(self,x):
            return self.a*x+self.b

    class Exponencial(Lineal):
        def __call__(self, x):
            return self.a * x ** self.b
    fs = 25
    L = Lineal(a,b)
    E = Exponencial(a,b)
    plt.figure(figsize=(10,8))
    sns.set(style='whitegrid')
    plt.plot(x,L(x),label=r'Lineal - $ax+b$',lw=3)
    plt.plot(x,E(x),label=r'Exponencial - $ax^b$',lw=3)
    plt.legend(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.title('a = {}, b = {}'.format(a,b),fontsize=fs)
    plt.tight_layout()
    plt.savefig('ej5-6.pdf')
    plt.show()

def ej7_8_9():

    from p0_lib import circunferencia as circle
    print(circle.pi())
    print(circle.area(2))
    from p0_lib.circunferencia import pi,area
    print(pi())
    print(area(2))
    print( circle.pi is pi )
    print( circle.area is area )

    import p0_lib
    from p0_lib import rectangulo
    from p0_lib.circunferencia import pi, area
    from p0_lib.elipse import area
    from p0_lib.rectangulo import area as area_rect

def ej10_11(): #contour y contourf en el mismo ej

    n = 10
    def f(x, y):
        return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

    plt.figure(figsize=(8,8),dpi=80)
    x = np.linspace(-3, 3, 4*n)
    y = np.linspace(-3, 3, 3*n)
    X, Y = np.meshgrid(x, y)
    plt.imshow(f(X,Y),cmap='bone',origin='lower')
    plt.xticks(())
    plt.yticks(())
    cbar = plt.colorbar(shrink=.5)
    cbar.ax.tick_params(labelsize=fs)
    plt.tight_layout()
    #plt.savefig("ej10.pdf")
    plt.show()
    

    plt.close()

    n=256

    plt.figure(figsize=(10,8),dpi=80)
    x = np.linspace(-3, 3, 4*n)
    y = np.linspace(-3, 3, 3*n)
    X, Y = np.meshgrid(x, y)
    C = plt.contourf(X,Y,f(X,Y),cmap=plt.cm.hot)
    C = plt.contour(X,Y,f(X,Y),colors='black',linewidths=3)
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()
    plt.clabel(C,fontsize=fs)#,manual=True)
    #plt.savefig("ej11.pdf")
    plt.show()
    
def ej12(n):
    
    X=np.random.normal(0,1,n)
    Y=np.random.normal(0,1,n)
    T = np.arctan2(Y,X)

    plt.figure(figsize=(10,8),dpi=80)
    plt.scatter(X, Y,c=T,alpha=0.5,cmap='jet', s=100)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()
    plt.savefig("ej12.pdf")
    plt.show()
    
def ej13(niter,step,N,maxVel,maxDist,size=1):
    class R2:

        def __init__(self, x=0, y=0):
            self.x = x
            self.y = y

        def __add__(self, other):
            return R2(self.x + other.x, self.y+other.y)

        def __sub__(self, other):
            return R2(self.x-other.x,self.y - other.y)

        def __mul__(self,scalar):
            return R2(self.x*scalar, self.y*scalar)

        def __abs__ (self):
            return np.sqrt(self.x**2 + self.y**2)

        def __truediv__(self,other):
            return R2(self.x/other, self.y/other)

        def __str__(self):
            return "({},{})".format(self.x,self.y)

    class Pez:

        def __init__(self, r, v):
            self.pos = r
            self.vel = v

        def move(self, t):
            self.pos = self.pos + self.vel*t

        def vchange( self, v, vmax, N ):
            self.vel = self.vel + v

            tita = np.arctan(self.vel.y/self.vel.x)

            if abs(self.vel) > vmax:
                self.vel.x = vmax * np.cos(tita)
                self.vel.y = vmax * np.sin(tita)

            if ( self.pos.x > N-1 ):
                self.vel.x = -abs(self.vel.x) ## Lo hago rebotar si llega a la pared
            if ( self.pos.x < 1 ):
                self.vel.x = abs(self.vel.x)
            if (self.pos.y > N-1 ):
                self.vel.y = -abs(self.vel.y)
            if (self.pos.y <  1):
                self.vel.y = abs(self.vel.y)

        def __str__(self):
            return "r={} , v={}".format(self.pos.__str__(),self.vel.__str__())

    class Cardumen:

        def __init__(self, size): #size = cantidad de peces
            self.size = size

        def initialize(self, N, maxVel, maxDist):
            self.maxDist = maxDist
            self.N = N
            self.maxVel = maxVel
            self.peces = [ Pez( R2( np.random.uniform(1,self.N-1) , np.random.uniform(1,self.N-1) ), 
                                R2( np.random.uniform(-self.maxVel,self.maxVel), np.random.uniform(-self.maxVel,self.maxVel) ) ) 
                                for i in range(self.size) ]

        def doStep(self,t):

            #Movimiento
            for i in range(self.size):
                self.peces[i].move(t)

            #Centro de masa y diferencia de velocidad 1
            posmean = np.mean( [ [ p.pos.x, p.pos.y ] for p in self.peces ], axis=0 )

            self.centromasa = R2 ( posmean[0],  posmean[1] )

            v1 = [ (self.centromasa-p.pos)/8 for p in self.peces ]

            #Diferencia de velocidad 2
            v2 =[ np.sum( [ (p.pos - t.pos)/abs(p.pos - t.pos) if ( 0 < abs(p.pos - t.pos) < self.maxDist ) 
                            else R2(0,0) for t in self.peces 
                            ] ) for p in self.peces ]

            #Velocidad media y diferencia de velocidad 3
            vmean = np.mean( [ [ p.vel.x, p.vel.y ] for p in self.peces ], axis=0 )

            self.vmedia = R2 ( vmean[0], vmean[1] )

            v3 = [ (self.vmedia - x.vel )/8 for x in self.peces]

            #Cambio de velocidad
            for i in range(self.size):
                self.peces[i].vchange( ( v3[i] + v2[i] + v1[i] ) , self.maxVel, self.N)


        def __str__(self):

            return str([ str(i) + ') ' + self.peces[i].__str__() + "\n" for i in range(self.size) ])

        def plot(self):
            p = np.array([ [ a.pos.x, a.pos.y] for a in self.peces])
            v = np.array([ [ a.vel.x, a.vel.y] for a in self.peces])
            plt.xlim=(0,self.N)
            plt.ylim=(0,self.N)
            return [plt.quiver(p[::,0],p[::,1],v[::,0], v[::,1])]

    c = Cardumen(size)
    c.initialize(N,maxVel,maxDist)
    gif = []
    fig = plt.figure(figsize=(6,6))

    #Las cosas comentadas son las necesarias para crear un mp4

    for i in range(niter):
        c.doStep(step)
        #gif.append(c.plot())

    ani = animation.ArtistAnimation( fig, gif, interval=200*step, repeat_delay=1000, blit=False)

    #fname = "step={}-maxVel={}-maxDis={}.mp4".format(step,maxVel,maxDist)

    #ani.save(fname) 

def ej14(N,l):

    def cumpleaños(m): #funcion que crea un vector de cumpleaños y comprueba si hay 2 que cumplan el mismo dia
        l = [ np.random.randint(365) for i in range(m) ]
        for i in range(m):
            if ( l.count(l[i]) > 1 ):
                return 1
        return 0

    def probabilidad(m,N): #calcula la probabilidad iterando en varios vectores
        k = 0
        for i in range(N):
            k += cumpleaños(m)
        return k/N*100

    def lista(N): #devuelve una lista con las probabilidades
       return ["{} personas: ".format(i*10) + str(round(probabilidad(i*10,N),3)) +'%' for i in range(1,l) ]

    print(lista(N))

def ej15(minV,maxV):
    class Noiser:

        def __init__(self,max,min):
            self.maxV = max
            self.minV = min

        def __call__(self,x):
            return x + np.random.uniform(self.minV,self.maxV)

    f = Noiser(minV,maxV)
    l = [i+0.0 for i in range(10)]
    f2 = np.vectorize(f)
    #f(l)
    #f2(l)
    #l = np.array(l) Para probar diferentes posibilidades
    #f(l)
    #f2(l)

if __name__ == "__main__":

    print('\nEjercicio 1')
    ej1()

    print('\nEjercicio 2')
    ej2()

    print('\nEjercicio 3')
    a, b, c = -1, 2, 3
    ej3(a,b,c)

    print('\nEjercicio 4')
    ej4(a,b,c)

    print('\nEjercicio 5-6')
    a, b= -2, 2
    x = np.arange(-5,5,0.1)
    ej5_6(a,b,x)

    print('\nEjercicio 7-8-9')
    ej7_8_9()

    print('\nEjercicio 10-11')
    ej10_11()

    print('\nEjercicio 12')
    n = 1000
    ej12(n)

    print('\nEjercicio 13') 
    niter = 200 #ver como imprimir en la funcion, hace un gif
    step = 0.5
    maxVel = 5
    maxDist = 5
    N = 40
    size = 16
    ej13(niter,step,N,maxVel,maxDist,size)

    print('\nEjercicio 14')
    N = 1000
    l = 7 #si es hasta grupos de x personas, poner x/10+1
    ej14(N,l)

    print('\nEjericio 15')
    minV = 0
    maxV = 1
    ej15(minV,maxV) #para probar ver el codigo

