import dearpygui.dearpygui as dpg
import numpy as np
import torch

class GUI:
    def __init__(self, camera, height, width, render_fn):
        self.camera = camera
        self.H = height
        self.W = width
        self.render_fn = render_fn
        self.buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.sampling_rate = 1
        dpg.create_context()
        self.__initialize_window()
        self.__initialize_GUI()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.render_frame()

    def __del__(self):
        dpg.destroy_context()

    def render_frame(self):
        ray_o, ray_d = self.camera.get_rays(device="cuda")
        rendered_out = self.render_fn(ray_o, ray_d)
        self.buffer = rendered_out['rgb']
        dpg.set_value("_texture", self.buffer)
        dpg.set_value("_camera_pos", self.camera.get_pose_text())

    def __initialize_window(self):
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        with dpg.window(tag="_window", width = self.W, height = self.H):
            dpg.add_image("_texture")

        dpg.set_primary_window("_window", True)

        def callback_rotate(sender, data):

            if not dpg.is_item_focused("_window"):
                return
            self.camera.rotate(data[1], data[2])

        def callback_scale(sender, data):
            if not dpg.is_item_focused("_window"):
                return
            self.camera.scale(data)

        def callback_pan(sender, data):
            if not dpg.is_item_focused("_window"):
                return
            self.camera.pan(data[1], data[2])

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_pan)

        dpg.create_viewport(title='Visualizer', width=self.W, height=self.H, resizable=False)

        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_window", theme_no_padding)



    def __initialize_GUI(self, opt=dict()):
        W = opt.get("width", 400)
        H = opt.get("height", 250)
        with dpg.window(label="Control", tag="_controls", width=W, height=H):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            with dpg.group(horizontal=True):
                dpg.add_text("Camera Pose: ")
                dpg.add_text("", tag="_camera_pos")

    def render(self):
        while dpg.is_dearpygui_running():
            self.render_frame()
            dpg.render_dearpygui_frame()
