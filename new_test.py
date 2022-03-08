from accelRF.visualizer.gui import GUI
from accelRF.visualizer.camera import OrbitCamera
import torch

def render_fn(ray_e, ray_d):
    corners = [
        [-0.5,0,0],
        [0.5,0,0],
        [0,0.8660254038,0]
      ]
    corners = torch.tensor(corners).float().cuda()
    min_t = 0.001
    tile_shape = (ray_d.shape[0], ray_d.shape[1], 1)
    abc = torch.tile(corners[0] - corners[1], tile_shape)
    def_ = torch.tile(corners[0] - corners[2], tile_shape)
    jkl = corners[0][None, None, :] - ray_e

    def_x_ghi = torch.cross(def_, ray_d)
    M = abc[:, :, 0] * def_x_ghi[:, :, 0] + \
        abc[:, :, 1] * def_x_ghi[:, :, 1] + \
        abc[:, :, 2] * def_x_ghi[:, :, 2]

    abc_x_jkl = torch.cross(abc, jkl)

    time =  - (def_[:, :, 0] * abc_x_jkl[:, :, 0] + \
            def_[:, :, 1] * abc_x_jkl[:, :, 1] + \
            def_[:, :, 2] * abc_x_jkl[:, :, 2] ) / M

    gamma = (ray_d[:, :, 0] * abc_x_jkl[:, :, 0] + \
             ray_d[:, :, 1] * abc_x_jkl[:, :,1] + \
             ray_d[:, :, 2] * abc_x_jkl[:, :, 2] ) / M

    beta = (jkl[:, :, 0] * def_x_ghi[:, :, 0] + \
            jkl[:, :, 1] * def_x_ghi[:, :, 1] + \
            jkl[:, :, 2] * def_x_ghi[:, :, 2] ) / M

    t = time
    t[(time < min_t) | \
      (gamma < 0) | (gamma > 1) | \
      (beta < 0) | (beta > 1) | \
      (beta + gamma > 1)] = 0
    t = t[:,:,None].repeat(1, 1, 3)
    return {'rgb': t.cpu().numpy()}

width, height = 720, 480
camera = OrbitCamera(height, width)
gui = GUI(camera, height, width, render_fn)

gui.render()
