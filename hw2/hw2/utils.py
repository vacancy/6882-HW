import math
import matplotlib.pyplot as plt

def display_image(img, title=None):
    """Render a figure inline
    """
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(img)
    _ = plt.axis('off')
    # plt.show()

def get_robot_pos(state, real=False):
    for obj, pos in state:
        if obj == 'robot0':
            if real: return pos
            return ((pos[0]+0.5)*50, (pos[1]+0.5)*50)

def draw_line(state1, state2, linewidth=5, color='#e74c3c'):
    point1 = get_robot_pos(state1)
    point2 = get_robot_pos(state2)
    plt.plot([point1[1], point2[1]], [point1[0], point2[0]], linewidth=linewidth, color=color)

def generate_color_wheel(original_color, size):

    def hex2int(hex1):
        return int('0x'+str(hex1),0)

    def int2hex(int1):
        hex1 = str(hex(int1)).replace('0x','')
        if len(hex1) == 1:
            hex1 = '0'+hex1
        return hex1

    def hex2ints(original_color):
        R_hex = original_color[0:2]
        G_hex = original_color[2:4]
        B_hex = original_color[4:6]
        R_int = hex2int(R_hex)
        G_int = hex2int(G_hex)
        B_int = hex2int(B_hex)
        return R_int, G_int, B_int

    def ints2hex(R_int, G_int, B_int):
        return '#'+int2hex(R_int)+int2hex(G_int)+int2hex(B_int)

    def portion(total, size, index):
        return total + round((225-total) / size * index)

    def gradients(start, end, size, index):
        return start + round((end-start) / size * index)

    color_wheel = []

    ## for experience replay, find all the colors between two colors
    if len(original_color) == 2:

        color1, color2 = original_color
        R1_int, G1_int, B1_int = hex2ints(color1.replace('#',''))
        R2_int, G2_int, B2_int = hex2ints(color2.replace('#',''))
        for index in range(size):
            color_wheel.append(ints2hex(
                gradients(R1_int, R2_int, size, index),
                gradients(G1_int, G2_int, size, index),
                gradients(B1_int, B2_int, size, index)
            ))

    ## for RL, the color of different shades symbolizes frequency
    else:

        R_int, G_int, B_int = hex2ints(original_color.replace('#',''))

        seq = list(range(size))
        seq.reverse()
        for index in seq:
            color_wheel.append(ints2hex(
                portion(R_int, size, index),
                portion(G_int, size, index),
                portion(B_int, size, index)
            ))

    return color_wheel

def initializee_color_wheel(color_density=None):

    COLOR_WHEEL = []

    ## rainbow color of the material UI style
    colors = ['#F44336','#E91E63','#9C27B0','#673AB7', '#3F51B5', '#2196F3', '#03A9F4', '#00BCD4', '#009688', '#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FFC107', '#FF9800', '#FF5722']
    COLOR_DENSITY = math.ceil(color_density/(len(colors)-1))
    for i in range(1,len(colors)):
        COLOR_WHEEL += generate_color_wheel((colors[i-1],colors[i]), COLOR_DENSITY)

    return COLOR_WHEEL

def extract_path(current_node):
    path = []
    while current_node.parent != None:
        path.append(current_node.state)
        current_node = current_node.parent
    path.append(current_node.state)

    path.reverse()
    draw_trace(path)
    return len(path)

def draw_trace(trace):
    """ plot the path on map """
    length = len(trace)
    color_wheel = initializee_color_wheel(length)
    for index in range(1, length):
        draw_line(trace[index], trace[index-1], linewidth=int(10 - (10 - 2) * index / length),
                  color=color_wheel[index])
    # plt.show()