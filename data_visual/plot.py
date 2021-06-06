# This code was used from: https://pythonprogramming.net/live-graphs-matplotlib-tutorial/ (whole file)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
# Make sure to empty data.txt before each training visualization 
style.use('grayscale')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open('data_visual/data.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:# Read lines from data.txt
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    ax1.clear()
    ax1.set_title("Loss with respect to epochs")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss/Error")
    ax1.plot(xs, ys)

ani = animation.FuncAnimation(fig, animate, interval=500)
plt.show()