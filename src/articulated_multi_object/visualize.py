import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib import animation, rc

colors = dict()
colors[0] = [0.2, 0.2, 0.2]
colors[1] = [0, 0, 1]
colors[2] = [0, 1, 0]
colors[3] = [1, 1, 0]
colors[4] = [0, 1, 1]
colors[5] = [1, 0, 1]
colors[6] = [0, 0.5, 0]
colors[7] = [0, 0, 0.5]
colors[8] = [0.5, 0.5, 0]
colors[9] = [0, 0.5, 0.5]
colors[10] = [0.5, 0, 0.5]
colors[11] = [0.5, 1, 0]
colors[12] = [0, 0.5, 1]
colors[13] = [1, 1, 0.5]
colors[14] = [0.5, 1, 1]
colors[15] = [1, 0.5, 1]


def visualize_PP(obj_shapes, edges, traj_real, traj_pred, filename):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    plt.close()
    titles = ['Real', 'Predicted']
    for ax_ind in range(2):
        ax[ax_ind].set_xlim((-1.5, 0.3))
        ax[ax_ind].set_ylim((-0.9, 0.9))
        ax[ax_ind].set_title(titles[ax_ind])

    obj_list = dict()
    obj_list[0] = []
    obj_list[1] = []

    edge_list = dict()
    edge_list[0] = []
    edge_list[1] = []

    for ax_ind in range(2):
        for obj_index, obj in enumerate(obj_shapes):
            c = colors[1+obj_index]
            if obj[1] == 0:
                circle = patches.Circle((0, 0), obj[-1]/2, color=c, ec='black')
                obj_txt = ax[ax_ind].text(
                    0, 0, str(obj_index), color='black', fontsize=12, weight='bold')
                obj_list[ax_ind].append((0, circle, obj_txt))
            else:
                patch_mini_list = list()
                for k in [-1, 1]:
                    for l in [-1, 1]:
                        patch = patches.Rectangle(
                            (0, 0), k*obj[-2]/2, l*obj[-1]/2, ec='black', fc=c)
                        patch_mini_list.append(patch)
                obj_txt = ax[ax_ind].text(
                    0, 0, str(obj_index), color='black', fontsize=12, weight='bold')
                obj_list[ax_ind].append((1, patch_mini_list, obj_txt))
        edg = edges
        cnt = 0
        for i in range(obj_shapes.shape[0]):
            for j in range(obj_shapes.shape[0]):
                if i != j:
                    if edg[cnt, 1] == 1:
                        c = 'black'
                    elif edg[cnt, 2] == 1:
                        c = 'purple'
                    elif edg[cnt, 3] == 1:
                        c = 'brown'
                    else:
                        cnt = cnt+1
                        continue
                    cnt = cnt+1
                    edge, = ax[ax_ind].plot([], [], color=c, lw=3)
                    edge_list[ax_ind].append((edge, (i, j)))
    time_txt = ax[0].text(-1.4, 0.8, str(0), color='black',
                          fontsize=12, weight='bold')

    def init():
        for ax_ind in range(2):
            for obj in obj_list[ax_ind]:
                if obj[0] == 0:
                    ax[ax_ind].add_patch(obj[1])
                else:
                    for patch in obj[1]:
                        ax[ax_ind].add_patch(patch)

        return []

    def animate(i):
        for ax_ind in range(2):
            if ax_ind == 0:
                positions = traj_pred[i]
            else:
                positions = traj_real[i]
            for obj_index, obj in enumerate(obj_list[ax_ind]):
                if obj[0] == 0:
                    obj[1].center = (positions[obj_index, 0],
                                     positions[obj_index, 1])
                else:
                    for patch in obj[1]:
                        patch.xy = (positions[obj_index, 0],
                                    positions[obj_index, 1])
                        patch.angle = np.rad2deg(positions[obj_index, 2])
            for edge in edge_list[ax_ind]:
                edge_line = edge[0]
                edge_info = edge[1]
                obj1 = edge_info[0]
                obj2 = edge_info[1]
                xx = [positions[obj1, 0], positions[obj2, 0]]
                yy = [positions[obj1, 1], positions[obj2, 1]]
                edge_line.set_data(xx, yy)
            for obj_index, obj in enumerate(obj_list[ax_ind]):
                obj[2].set_position((positions[obj_index, 0],
                                     positions[obj_index, 1]))
            time_txt.set_text(str(i))
        return []
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=int(traj_real.shape[0]),
                                   interval=50, blit=True)
    anim.save(filename+'.gif', writer='imagemagick', fps=30)
    return anim


def visualize_BR(obj_shapes, predicted_edges, traj_real, filename):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    plt.close()
    titles = ['Belief Regulation']
    ax.set_xlim((-1.5, 0.3))
    ax.set_ylim((-0.9, 0.9))
    ax.set_title(titles)

    obj_list = []

    edge_list = []

    for obj_index, obj in enumerate(obj_shapes):
        c = colors[1+obj_index]
        if obj[1] == 0:
            circle = patches.Circle((0, 0), obj[-1]/2, color=c, ec='black')
            obj_txt = ax.text(
                0, 0, str(obj_index), color='red', fontsize=12, weight='bold')
            obj_list.append((0, circle, obj_txt))
        else:
            patch_mini_list = list()
            for k in [-1, 1]:
                for l in [-1, 1]:
                    patch = patches.Rectangle(
                        (0, 0), k*obj[-2]/2, l*obj[-1]/2, ec='black', fc=c)
                    patch_mini_list.append(patch)
            obj_txt = ax.text(0, 0, str(obj_index),
                              color='red', fontsize=12, weight='bold')
            obj_list.append((1, patch_mini_list, obj_txt))
    edg = predicted_edges[0]
    cnt = 0
    for i in range(obj_shapes.shape[0]):
        for j in range(obj_shapes.shape[0]):
            if i != j:
                max_edge = np.argmax(edg[cnt])
                alp = 1  # edg[cnt, max_edge]
                if max_edge == 0:
                    alp = 0
                    c = 'white'
                elif max_edge == 1:
                    c = 'black'
                elif max_edge == 2:
                    c = 'purple'
                elif max_edge == 3:
                    c = 'brown'
                cnt = cnt+1
                edge, = ax.plot([], [], color=c, alpha=alp, lw=3)
                edge_list.append((edge, (i, j)))
    time_txt = ax.text(-1.4, 0.8, str(0), color='black',
                       fontsize=12, weight='bold')

    def init():
        for obj in obj_list:
            if obj[0] == 0:
                ax.add_patch(obj[1])
            else:
                for patch in obj[1]:
                    ax.add_patch(patch)

        return []

    def animate(i):
        positions = traj_real[i]
        for obj_index, obj in enumerate(obj_list):
            if obj[0] == 0:
                obj[1].center = (positions[obj_index, 0],
                                 positions[obj_index, 1])
            else:
                for patch in obj[1]:
                    patch.xy = (positions[obj_index, 0],
                                positions[obj_index, 1])
                    patch.angle = np.rad2deg(positions[obj_index, 2])
        for edge_ind, edge in enumerate(edge_list):
            edge_line = edge[0]
            edge_info = edge[1]
            obj1 = edge_info[0]
            obj2 = edge_info[1]
            xx = [positions[obj1, 0], positions[obj2, 0]]
            yy = [positions[obj1, 1], positions[obj2, 1]]
            edg = predicted_edges[i, edge_ind]
            max_edge = np.argmax(edg)
            alp = 1  # edg[max_edge]
            if max_edge == 0:
                alp = 0
                c = 'white'
            elif max_edge == 1:
                c = 'black'
            elif max_edge == 2:
                c = 'purple'
            elif max_edge == 3:
                c = 'brown'
            edge_line.set_color(c)
            edge_line.set_alpha(alp)
            edge_line.set_data(xx, yy)
        for obj_index, obj in enumerate(obj_list):
            obj[2].set_position((positions[obj_index, 0]-0.025,
                                 positions[obj_index, 1]-0.025))
        time_txt.set_text(str(i))
        return []
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=int(traj_real.shape[0]),
                                   interval=50, blit=True)
    anim.save(filename+'.gif', writer='imagemagick', fps=15)
    return anim
