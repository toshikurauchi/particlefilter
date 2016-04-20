#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches

print("Loadmap called")

plt = None

def loadmap(plt_nb):
    print("setting plt")
    global plt
    plt = plt_nb
    if !plt:
        print("plt is None")

back_color = "black"
colors     = ['red', 'green', 'cyan', 'yellow']
width, height = 480, 480


def draw_map(map, particles = None):
    global plt
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlim=[0, width], ylim=[0, height]) # Or use "ax.axis([x0,x1,y0,y1])"

    rect = mpatches.Rectangle(
        (0, 0), 50, 50,
        facecolor = choice(colors),
        edgecolor = back_color
    )
    ax.add_artist(rect)


    fig.canvas.draw()

    #def update():

    plt.imshow(np_image, cmap='Greys_r')
