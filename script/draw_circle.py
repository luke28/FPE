from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt

def main():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    #ell1 = Ellipse(xy=(0.0,0.0),width=4,height=8,angle=30.0,facecolor='white',alpha=0.3)
    cir1 = Circle(xy=(0,0),radius=2, fill = False, ec='r', alpha = 1)
    #cir2 = Circle(xy=(1.0,2.0), radius = 3, fill = False, alpha=1)
    #ax.add_patch(ell1)
    ax.add_patch(cir1)
    #ax.add_patch(cir2)
    #x1 = [1, 2, 3]
    #y1 = [4, 3, 1]
    #ax.plot(x1,y1,'ro')
    #x2, y2 = 1,1
    #ax.plot(x2,y2,'ro')
    #x3, y3 = 2,2
    #ax.plot(x3,y3,'bx')
    plt.axis('scaled')
    #ax.set_xlim(-10,10)
    #ax.set_ylim(-10,10)
    #plt.axis([-10,10,-10,10])
    plt.axis('equal')
    plt.show()

if __name__=='__main__':
   main()
   #plt.plot([1,2,3,4], [10,2,13,1], 'ro')
   #plt.show()
