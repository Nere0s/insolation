import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import xarray as xr
import matplotlib.animation as animation
import numpy as np

def sun_pos(td = .25, ty = .25):
    tilt = .2 + .05 * np.cos(2 * np.pi * ty)

    _x = np.sin(2 * np.pi * td) * np.sin(tilt * np.pi)
    _y = -np.cos(2 * np.pi * td)
    _z = np.sin(2 * np.pi * td) * np.cos(tilt * np.pi)

    return (_x, _y, _z)

def setup_data(Nx=100, Ny=100, Ntd=24, Nty=12):

    x_axis  = np.linspace(-1, 1, Nx)
    y_axis  = np.linspace(-1, 1, Ny)
    td_axis = np.linspace( 0, .5, Ntd)
    ty_axis = np.linspace( 0, 1., Nty, endpoint=False)

    coords = {'quantity': ['dist', 'insol'], 'x': x_axis, 'y': y_axis, 't_day': td_axis, 't_year': ty_axis}

    xx, yy = np.meshgrid(x_axis, y_axis)

    data = xr.DataArray(np.full((2, Nx, Ny, Ntd, Nty), np.nan), dims=['quantity', 'x', 'y', 't_day', 't_year'], coords = coords)

    for t_y in data.coords['t_year'].values:
        for t_d in data.coords['t_day'].values:
            pos = sun_pos(t_d, t_y)

            dist = np.sqrt( (xx - pos[0])**2 + (yy - pos[1])**2 + pos[2]**2)
            insol = 1/dist**2 * np.exp(-dist) * np.arcsin(pos[2] / dist)
            #insol = 1/dist**2 * np.arcsin(pos[2] / dist)

            data.loc[{'t_day': t_d, 't_year': t_y, 'quantity': 'dist'}]  = dist
            data.loc[{'t_day': t_d, 't_year': t_y, 'quantity': 'insol'}] = insol

    return data

def make_frames(data):

    zoom_area = .2

    N_d = len(data.coords['t_day'])
    N_y = len(data.coords['t_year'])
    N_tot = N_d * N_y

    ###########################################################################
    # setup figure
    ###########################################################################

    fig = plt.figure(figsize=[11, 12])

    axs = []
    axs.append(plt.subplot2grid((4, 3), (0, 0), projection='3d', rowspan=2, colspan=2))
    axs.append(plt.subplot2grid((4, 3), (0, 2)))
    axs.append(plt.subplot2grid((4, 3), (1, 2)))
    axs.append(plt.subplot2grid((4, 3), (2, 0)))
    axs.append(plt.subplot2grid((4, 3), (3, 0)))
    axs.append(plt.subplot2grid((4, 3), (2, 1)))
    axs.append(plt.subplot2grid((4, 3), (3, 1)))
    axs.append(plt.subplot2grid((4, 3), (2, 2)))
    axs.append(plt.subplot2grid((4, 3), (3, 2)))

    for i, ax in enumerate(axs):
        if i == 0:
            ax.set_xticks([-1, -.5, 0, .5, 1])
            ax.set_yticks([-1, -.5, 0, .5, 1])
            ax.set_zticks([0, .5, 1, 1.5])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            axs[0].set_zlim(0, 1.65)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if i%2 != 0 or i == 0:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        else:
            ax.set_xlim(-zoom_area, zoom_area)
            ax.set_ylim(-zoom_area, zoom_area)

    plt.tight_layout(w_pad=4, h_pad=2, pad=2)

    # xx, yy = np.meshgrid(data.coords['x'], data.coords['y'])

    #######################################################################
    # 3D plot
    #######################################################################

    title_3d = axs[0].set_title('', pad=-1)

    # plot ground
    axs[0].plot([ zoom_area,  zoom_area], [ zoom_area, -zoom_area], -.02, color='black', alpha=.5)
    axs[0].plot([ zoom_area, -zoom_area], [-zoom_area, -zoom_area], -.02, color='black', alpha=.5)
    axs[0].plot([-zoom_area, -zoom_area], [-zoom_area,  zoom_area], -.02, color='black', alpha=.5)
    axs[0].plot([-zoom_area,  zoom_area], [ zoom_area,  zoom_area], -.02, color='black', alpha=.5)

    # project position on ground
    sun_path_proj_z, = axs[0].plot([0, 0], [0, 0], [0, 0], color='grey')
    sun_path_proj_y, = axs[0].plot([0, 0], [0, 0], [0, 0], color='grey')
    sun_path_proj_x, = axs[0].plot([0, 0], [0, 0], [0, 0], color='grey')
    sun_pos_proj, = axs[0].plot([0], [0], [0], color='black', linestyle='dotted')
    sun_pos_proj_x, = axs[0].plot([0], [0], [0], color='black', linestyle='none', marker='.')
    sun_pos_proj_y, = axs[0].plot([0], [0], [0], color='black', linestyle='none', marker='.')
    sun_pos_proj_z, = axs[0].plot([0], [0], [0], color='black', linestyle='none', marker='.')

    # plot position of the sun
    sun_path_3d, = axs[0].plot([0, 0], [0, 0], [0, 0], color='yellow')
    sun_pos_3d,  = axs[0].plot([0], [0], [0], color='orange', linestyle='none', marker='.', markeredgecolor='black', markersize=15)

    #######################################################################
    # distance grid
    #######################################################################

    axs[1].set_title(r'distance to sun $[L]$')
    axs[1].set_ylabel('full view')
    axs[2].set_ylabel('zoomed view')

    # full view
    dist_grid_full = axs[1].imshow(data.sel(quantity='dist').isel({'t_day':0, 't_year':0}), cmap='gray', vmin=0, vmax=2, extent=(-1,1,1,-1))
    fig.colorbar(dist_grid_full, ax=axs[1])

    map_border = patches.Rectangle((-zoom_area,-zoom_area),2*zoom_area,2*zoom_area,linewidth=1,edgecolor='lightgrey',facecolor='none')
    axs[1].add_patch(map_border)

    # zoom view
    dist_grid_zoom = axs[2].imshow(data.sel(quantity='dist').isel({'t_day':0, 't_year':0}), cmap='gray', vmin=.8, vmax=1.2, extent=(-1,1,1,-1))
    fig.colorbar(dist_grid_zoom, ax=axs[2])

    #######################################################################
    # insolation grid
    #######################################################################

    axs[3].set_title(r'insolation $[M \cdot T^{-3}]$')
    axs[3].set_ylabel('full view')
    axs[4].set_ylabel('zoomed view')

    # full view
    inso_grid_full = axs[3].imshow(data.sel(quantity='insol').isel({'t_day':2, 't_year':0}), cmap='inferno', vmin=0, vmax=1.4, extent=(-1,1,1,-1))
    fig.colorbar(inso_grid_full, ax=axs[3])

    map_border = patches.Rectangle((-zoom_area,-zoom_area),2*zoom_area,2*zoom_area,linewidth=1,edgecolor='lightgrey',facecolor='none')
    axs[3].add_patch(map_border)

    # zoom view
    inso_grid_zoom = axs[4].imshow(data.sel(quantity='insol').isel({'t_day':2, 't_year':0}), cmap='inferno', vmin=0, vmax=.6, extent=(-1,1,1,-1))
    fig.colorbar(inso_grid_zoom, ax=axs[4])

    #######################################################################
    # daily mean insolation grid
    #######################################################################

    axs[5].set_title(r'daily mean insolation $[M \cdot T^{-3}]$')

    # full view
    mean_grid_full = axs[5].imshow(data.sel(quantity='insol').isel({'t_year':0}).mean(dim='t_day'), cmap='inferno', vmin=0, vmax=1, extent=(-1,1,1,-1))
    fig.colorbar(mean_grid_full, ax=axs[5])

    map_border = patches.Rectangle((-zoom_area,-zoom_area),2*zoom_area,2*zoom_area,linewidth=1,edgecolor='lightgrey',facecolor='none')
    axs[5].add_patch(map_border)

    # zoom view
    mean_grid_zoom = axs[6].imshow(data.sel(quantity='insol').isel({'t_year':0}).mean(dim='t_day'), cmap='inferno', vmin=.1, vmax=.3, extent=(-1,1,1,-1))
    fig.colorbar(mean_grid_zoom, ax=axs[6])

    #######################################################################
    # annual mean insolation grid
    #######################################################################

    axs[7].set_title(r'annual mean insolation $[M \cdot T^{-3}]$')

    # full view
    annual_grid_full = axs[7].imshow(data.sel(quantity='insol').mean(dim=['t_day','t_year']), cmap='inferno', vmin=0, vmax=1, extent=(-1,1,1,-1))
    fig.colorbar(annual_grid_full, ax=axs[7])

    map_border = patches.Rectangle((-zoom_area,-zoom_area),2*zoom_area,2*zoom_area,linewidth=1,edgecolor='lightgrey',facecolor='none')
    axs[7].add_patch(map_border)

    # zoom view
    annual_grid_zoom = axs[8].imshow(data.sel(quantity='insol').mean(dim=['t_day','t_year']), cmap='inferno', vmin=.1, vmax=.3, extent=(-1,1,1,-1))
    fig.colorbar(annual_grid_zoom, ax=axs[8])


    ###########################################################################
    # update loop
    ###########################################################################

    for i in range(N_tot):
        file_name = '.frames/frame_{:04d}.png'.format(i)

        i_d = i % N_d
        i_y = i // N_d

        print('t_d {:2d}/{:2d} | t_y {:2d}/{:2d}'.format(i_d+1, N_d, i_y+1, N_y), end='\r')

        t_d = data.coords['t_day'][i_d]
        t_y = data.coords['t_year'][i_y]
        pos = sun_pos(t_d, t_y)
        path = np.array([sun_pos(_t_d, t_y) for _t_d in np.linspace(0., .5)])
        #######################################################################
        # 3D plot
        #######################################################################

        title_3d.set_text('day fraction {:3d}% | year fraction {:3d}%'.format(int(i_d / (N_d-1) * 100), int(i_y / N_y * 100)))

        # plot position of the sun
        sun_path_3d.set_data(*path[:,:2].transpose())
        sun_path_3d.set_3d_properties(path[:,2])
        sun_pos_3d.set_data(*pos[:2])
        sun_pos_3d.set_3d_properties(pos[2])

        # projection
        sun_pos_proj.set_data([pos[0], pos[0]], [pos[1], pos[1]])
        sun_pos_proj.set_3d_properties([0, pos[2]])

        sun_pos_proj_x.set_data(-.999, pos[1])
        sun_pos_proj_x.set_3d_properties(pos[2])
        sun_pos_proj_y.set_data(pos[0], 1)
        sun_pos_proj_y.set_3d_properties(pos[2])
        sun_pos_proj_z.set_data(*pos[:2])
        sun_pos_proj_z.set_3d_properties(0)

        sun_path_proj_z.set_data(*path[:,:2].transpose())
        sun_path_proj_z.set_3d_properties(np.zeros_like(pos[2]))
        sun_path_proj_y.set_data(path[:,0].transpose(), np.full_like(path[:,0].transpose(), 1))
        sun_path_proj_y.set_3d_properties(path[:,2])
        sun_path_proj_x.set_data(np.full_like(path[:,1].transpose(), -1), path[:,1].transpose())
        sun_path_proj_x.set_3d_properties(path[:,2])

        #######################################################################
        # distance grid
        #######################################################################
        dist_grid_full.set_data(data.sel({'quantity':'dist', 't_day':t_d, 't_year':t_y}))
        dist_grid_zoom.set_data(data.sel({'quantity':'dist', 't_day':t_d, 't_year':t_y}))

        #######################################################################
        # insolation grid
        #######################################################################
        inso_grid_full.set_data(data.sel({'quantity':'insol', 't_day':t_d, 't_year':t_y}))
        inso_grid_zoom.set_data(data.sel({'quantity':'insol', 't_day':t_d, 't_year':t_y}))

        #######################################################################
        # mean insolation grid
        #######################################################################
        mean_grid_full.set_data(data.sel({'quantity':'insol', 't_year':t_y}).mean(dim='t_day'))
        mean_grid_zoom.set_data(data.sel({'quantity':'insol', 't_year':t_y}).mean(dim='t_day'))

        #######################################################################
        # draw and save
        #######################################################################

        plt.draw()
        plt.savefig(file_name)

    print()


data = setup_data()

make_frames(data)
