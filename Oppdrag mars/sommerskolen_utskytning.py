import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#Plotting the Earth and Mars orbits
alpha_input = input("Tast inn utskytningsvinkel: ")
alpha = float(alpha_input) #degrees (Angle by which should be ahead by)
Earth = plt.Circle((0,0), radius= 1.0,fill=False,color='blue')
Mars = plt.Circle((0,0), radius= 1.52,fill=False,color='brown')

#Moving Earth, Mars, and Spacecraft
patch_E = plt.Circle((0.0, 0.0),radius=0.04,fill=True,color='blue')
patch_M = plt.Circle((0.0, 0.0),radius=0.03,fill=True,color='brown')
patch_H = plt.Circle((0.0, 0.0),radius=0.01,fill=True,color='red')

def init():
 patch_E.center = (0.0,0.0)
 ax.add_patch(patch_E)
 patch_M.center = (0.0,0.0)
 ax.add_patch(patch_M)
 patch_H.center = (0.0,0.0)
 ax.add_patch(patch_H)
 return patch_E,patch_M,patch_H
 
def animate(i):
    # Earth
    x_E, y_E = patch_E.center
    x_E = np.cos((2*np.pi/365.2)*i)
    y_E = np.sin((2*np.pi/365.2)*i)
    patch_E.center = (x_E, y_E)

    # Mars
    x_M, y_M = patch_M.center
    x_M = 1.52*np.cos((2*np.pi/686.98)*i+(np.pi*alpha/180.))
    y_M = 1.52*np.sin((2*np.pi/686.98)*i+(np.pi*alpha/180.))
    patch_M.center = (x_M,y_M)

    # Hohmann
    Period = 516.0
    x_H = 1.26*(1.0 - 0.21**2)/(1.0 + 0.21*np.cos((2*np.pi/Period)*i))*np.cos((2*np.pi/Period)*i)
    y_H = 1.26*(1.0 - 0.21**2)/(1.0 + 0.21*np.cos((2*np.pi/Period)*i))*np.sin((2*np.pi/Period)*i)
    patch_H.center = (x_H,y_H)
    return patch_E,patch_M,patch_H

# Set up formatting for the movie files
#plt.rcParams[‘savefig.bbox’] = ‘tight’ # tight garbles the video!!!
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)

# Set up path, to guide eye
Period = 516.
x_H_B = 1.26*(1.0 - 0.21**2)/(1.0 + 0.21*np.cos((2*np.pi/Period*75)))*np.cos((2*np.pi/Period*75))
y_H_B = 1.26*(1.0 - 0.21**2)/(1.0 + 0.21*np.cos((2*np.pi/Period*75)))*np.sin((2*np.pi/Period*75))
x_H_C = 1.26*(1.0 - 0.21**2)/(1.0 + 0.21*np.cos((2*np.pi/Period*150)))*np.cos((2*np.pi/Period*150))
y_H_C = 1.26*(1.0 - 0.21**2)/(1.0 + 0.21*np.cos((2*np.pi/Period*150)))*np.sin((2*np.pi/Period*150))
x_H_D = 1.26*(1.0 - 0.21**2)/(1.0 + 0.21*np.cos((2*np.pi/Period*200)))*np.cos((2*np.pi/Period*200))
y_H_D = 1.26*(1.0 - 0.21**2)/(1.0 + 0.21*np.cos((2*np.pi/Period*200)))*np.sin((2*np.pi/Period*200))
x_H_M = 1.26*(1.0 - 0.21**2)/(1.0 + 0.21*np.cos((2*np.pi/Period*250)))*np.cos((2*np.pi/Period*250))
y_H_M = 1.26*(1.0 - 0.21**2)/(1.0 + 0.21*np.cos((2*np.pi/Period*250)))*np.sin((2*np.pi/Period*250))


def getImage(path):
   return OffsetImage(plt.imread(path, format="jpg"), zoom=.1)

fig, ax = plt.subplots(figsize=(10,8))
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

ax.plot(0,0,color='orange',marker='o',linestyle='',markersize=16,markerfacecolor='yellow',label='Sun')

ax.plot([],[],color='blue',linestyle='',marker='o',label='Earth')

ax.plot([],[],color='brown',linestyle='',marker='o',label='Mars')
ax.plot([],[],color='red',linestyle='',marker='o',label='spacecraft')
ax.plot(x_H_B,y_H_B,color='dimgray',marker ='p',markerfacecolor='dimgray',linestyle='',label='path')
ax.plot(x_H_C,y_H_C,color='dimgray',marker ='p',markerfacecolor='dimgray')
ax.plot(x_H_D,y_H_D,color='dimgray',marker ='p',markerfacecolor='dimgray')
ax.plot(x_H_M,y_H_M,color='dimgray',marker ='p',markerfacecolor='dimgray')
ax.add_patch(Earth)
ax.add_patch(Mars)
ax.set_xlabel('X [AU]',fontsize=12)
ax.set_ylabel('Y [AU]',fontsize=12)
ax.legend(loc='best',fontsize=12)
anim = animation.FuncAnimation(fig, animate,init_func=init,frames=260,interval=40,blit=True)
plt.axis('scaled') #Scale the plot in real time
plt.show()