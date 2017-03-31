from random import randint, choice
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
import random
from pf import Particle
from nav_msgs.msg import OccupancyGrid
from occupancy_field import OccupancyField
from helper_functions import angle_normalize, angle_diff



def convert_to_figure(xy_theta):
    """ 
        Converts a xy_theta to screen coordinates
    """
    pass


def nb_draw_map(mapa_numpy, particles = None, initial_position=False, pose=False, robot=False):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlim=[0, width], ylim=[0, height]) # Or use "ax.axis([x0,x1,y0,y1])"

    fig.canvas.draw()

    #def update():

    plt.imshow(mapa_numpy, cmap='Greys_r')
    if initial_position:
        draw_initial_pose(initial_pose,ax)
    if particles:
        nb_draw_particle_cloud(particles, ax)
    if pose:
        nb_draw_arrow(pose[0], pose[1], pose[2], ax, color='g', width=2, headwidth=6, headlength=6)
    if robot:
        nb_draw_robot(pose, ax, radius=robot_radius)
    
    return ax # Retornamos o contexto grafico caso queiram fazer algo depois

        
def draw_initial_pose(pose_xytheta, ax):
    """
        Metodo que desenha a pose inicial
        pose - um array que contem x, y e theta da pose inicial
        ax - um objeto do matplotlib
    """
    x = pose_xytheta[0]
    y = pose_xytheta[1]
    theta = pose_xytheta[2]
    l = 15
    #end_x = x + deltax
    #end_y = y + deltay
    nb_draw_arrow(x, y, theta, ax, l=l, color='r', width=2, headwidth=6, headlength=6)
    
def nb_draw_arrow(x, y, theta, ax, l = 15, color='y', headwidth=3.0, headlength=3, width=0.001):
    deltax = l*math.cos(theta)
    deltay = l*math.sin(theta)
    ax.arrow(x, y, deltax, deltay, head_width=headwidth, head_length=headlength, fc=color,  ec=color, width=width)

def nb_draw_particle_cloud(particles, ax):
    """
        Desenha o particle cloud
        particles - uma lista de objetos Particle
        ax - eixo
    """
    for p in particles:
        nb_draw_arrow(p.x, p.y, p.theta, ax, particle_size, color='b')

def normalize_particles():
    #global particle_cloud
    w_sum = 0
    for p in particle_cloud:
        w_sum+=p.w
    for p in particle_cloud:
        p.normalize(w_sum)

def update_robot_pose():
    """ Update the estimate of the robot's pose given the updated particles.
        There are two logical methods for this:
            (1): compute the mean pose
            (2): compute the most likely pose (i.e. the mode of the distribution)
    """
    # first make sure that the particle weights are normalized
    normalize_particles()

    # TODO: assign the lastest pose into self.robot_pose as a geometry_msgs.Pose object
    # Nota: neste notebook ao inves de usarmos um objeto Pose vou usar uma lista x, y, theta
    robot_pose = [0, 0, 0]
        
def nb_initialize_particle_cloud(xy_theta=None):
    """ Initialize the particle cloud.
        Arguments
        xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                  particle cloud around.  If this input is ommitted, the odometry will be used """
    if xy_theta == None:
        #xy_theta = convert_pose_to_xy_and_theta(.odom_pose.pose)
        pass
    global particle_cloud
    # TODO create particles
    particle_cloud = nb_create_particles(initial_pose)
        
    normalize_particles()
    update_robot_pose()    
    
def nb_create_particles(pose, var_x = 50, var_y = 50, var_theta = math.pi/3, num=30):
    """
        Cria num particulas
        situadas no intervalo x - var_x a x + var_x, y - var_x at'e y + var_y e theta - var_theta a theta + var_theta
    """
    particle_cloud = []
    s = pose
    for i in range(num):
        x = random.uniform(s[0] - var_x, s[0] + var_x)
        y = random.uniform(s[1] - var_x, s[1] + var_y)
        theta = random.uniform(s[2] - var_theta, s[2] + var_theta)
        p = Particle(x, y, theta, w=1.0) # A prob. w vai ser normalizada depois
        particle_cloud.append(p)
    return particle_cloud

def nb_draw_robot(position, ax, radius=10):
    from matplotlib.patches import Circle
    circle = Circle((position[0], position[1]), radius, facecolor='none',
                    edgecolor=(0.0, 0.8, 0.2), linewidth=2, alpha=0.7)
    ax.add_patch(circle)
    
def nb_create_ros_map(numpy_image):
    """
        Este notebook nao usa o service GetMap, portanto
        precisamos usar a imagem que foi lida e criar um OccupancyGrid
    """
    grid = OccupancyGrid()
    grid.info.resolution = 1
    w = numpy_image.shape[0]
    h = numpy_image.shape[1]    
    grid.info.width = w
    grid.info.height = h
    image_data = []
    for i in range(numpy_image.size):
        cell = 1.0 - (numpy_image[i//w][i%w]/255.0)
        if cell < 0.005:
            cell = 0
        image_data.append(cell)
    
    print("Occurences of zero",image_data.count(0))
    grid.data = image_data
    return grid

def nb_interp(min_a, max_a, a, dst_min, dst_max):
    """
        Funcao de interpolacao generica.
        min_a
        max_a
        a - valor na faixa de origem
        dst_min, dst_max - a faixa de destino
    """    
    return dst_min + ((a - min_a)/(max_a - min_a))*(dst_max - dst_min)

def nb_cria_occupancy_field_image(occupancy_field, numpy_image):
    occupancy_image = numpy_image.copy() # todo: extrair do occupancy_grid no futuro
    max_dist = max(occupancy_field.closest_occ.values())
    min_dist = min(occupancy_field.closest_occ.values())
    for i in range(occupancy_image.shape[1]):
        for j in range(occupancy_image.shape[0]):
            occupancy_image[i][j] = int(nb_interp(min_dist, max_dist, occupancy_field.get_closest_obstacle_distance(j,i), 0, 255))
    return occupancy_image
   



def nb_outside_image(x, y, img):
    if x > img.shape[1] or x < 0:
        return true
    if y > img.shape[0] or y < 0:
        return ture

def nb_found_obstacle(x, y, x0, y0, img):
    gray_value = 1.0 - img[x][y]/255.0
    if gray_value > free_thresh and gray_value < occupied_thresh:
        return math.sqrt( (x0 - x)**2 + (y0 - x)**2 )

        
    
def nb_find_discrete_line_versor(xa, ya, angle):
    m = math.tan(angle)
    delta = 50.0 # arbitrario
    xd = xa + delta*math.cos(angle)
    yd = ya + delta*math.sin(angle)
    deltay = yd - ya
    deltax = xd - xa
    v = [deltax, deltay]
    length = math.sqrt(deltax**2 + deltay**2)
    versor = [deltax/delta, deltay/delta]
    # Um pouco ineficiente mas 'garante' que nao pularemos celulas
    for i in range(len(versor)):
        versor[i]*=0.6
    return versor
    


def nb_simulate_lidar(robot_pose, angles, img):
    """
        Simula a leitura `real` do LIDAR supondo que o robot esteja na robot_pose e com sensores nos angulos angles
        
        Nao e' necessario fazer isso em seu projeto
        
        retorna uma lista de pontos de intersecao ou -1 se o sensor nao ler nada naquele angulo
        
    """
    a = angles.copy()
    theta = 2 # para ficar mais intuitivo
    
    #robot_pose[theta] = angle_normalize(robot_pose[theta])
    
    lidar_results = {}
    
    result_img = np.zeros(img.shape)
    result_img.fill(255) # Deixamos tudo branco
    
    x0 = robot_pose[0]
    y0 = robot_pose[1]
    
    for angulo in a:
        # Faz o angulo ser relativo ao robo
        ang = robot_pose[theta]+angulo
        #print("Angle ", ang)
        xa, ya = x0, y0
        x = xa
        y = ya
        vers = nb_find_discrete_line_versor(xa, ya, ang)
        #print("vers ", ang, "  " , vers)

        while (True):
            result_img[int(y), int(x)] = 0 # Marcamos o raio na imagem y,x porque numpy e' linha, coluna
            if nb_outside_image(int(x), int(y), img):
                # A imagem acabou, nao achamos nada
                lidar_results[ang] = -1   
                print("Outside at ",x ,"  ",y, "  for angle ", ang)
                break
            dist = nb_found_obstacle(int(y), int(x), y0, x0, img)
            if dist > -1:   
                # Achamos alguma coisa
                lidar_results[ang] = dist 
                #print("Hit for ",x,  "  ",y, "  for angle ", ang)                
                break
            # Keep going if none of the "ifs" has been triggered
            x += vers[0]
            y += vers[1]
            
    return lidar_results, result_img
