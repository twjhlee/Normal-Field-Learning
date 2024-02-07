import torch
import numpy as np
# from einops import rearrange
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import time
import warnings; warnings.filterwarnings("ignore")
import cv2
import configargparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GUI:
    @torch.no_grad()
    def __init__(self, wh):
        self.register_dpg()
        self.W, self.H = wh
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)

    def register_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title="Rendering", width=self.W, height=self.H, resizable=False)

        ## register texture ##
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture")

        ## register window ##
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")
        dpg.set_primary_window("_primary_window", True)

        def on_click_change_view(sender, on_click_callback):
            on_click_callback()

        ## control window ##
        with dpg.window(label="Control", tag="_control_window", width=200, height=150):
            dpg.add_button(label="Change View", tag="_button_changeview",
                            callback=on_click_change_view)
            dpg.add_separator()
            dpg.add_text('no data', tag='_log_time')
            # dpg.add_text('no data', tag='_log_position')

        # with dpg.handler_registry():
        #     dpg.add_mouse_drag_handler(
        #         button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
        #     )
        #     dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
        #     dpg.add_mouse_release_handler(
        #         button=dpg.mvMouseButton_Left, callback=on_left_mouse_release
        #     )
        #     dpg.add_key_down_handler(
        #         callback=on_key_down
        #     )

        ## Avoid scroll bar in the window ##
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        ## Launch the gui ##
        dpg.setup_dearpygui()
        # dpg.set_viewport_small_icon("assets/icon.png")
        # dpg.set_viewport_large_icon("assets/icon.png")
        dpg.show_viewport()

    def set_texture(self, img):
        dpg.set_value("_texture", img)
    # def render(self):
    #     while dpg.is_dearpygui_running():
    #         dpg.set_value("_texture", self.render_cam(self.cam))
    #         dpg.set_value("_log_time", f'Render time: {1000 * self.dt:.2f} ms')
    #         dpg.set_value("_log_position", f'Position: {self.cam.position - self.cam.center}')
    #         dpg.render_dearpygui_frame()


if __name__ == "__main__":
    with torch.no_grad():
        np.set_printoptions(precision=2)
        parser = configargparse.ArgumentParser()
        parser.add_argument("--ckpt", type=str, required=True,
                            help='specific weights npy file to reload for coarse network')
        parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                            choices=['TensorVMSplit', 'TensorCP', 'EgoNeRF'])
        parser.add_argument('--downsample', type=float, default=1.0)
        parser.add_argument('--exp_sampling', type=bool, default=False)
        parser.add_argument('--resampling', type=bool, default=False)
        parser.add_argument('--n_coarse', type=int, default=-1)
        parser.add_argument('--n_fine', type=int, default=-1)
        parser.add_argument('--N_samples', type=int, default=-1)
        parser.add_argument('--chunk_size', type=int, default=1024 * 8)
        args = parser.parse_args()

        downsample_gui = args.downsample
        # img_wh = [int(3940 / downsample_gui), int(1920 / downsample_gui)]
        img_wh = [int(2000 / downsample_gui), int(1000 / downsample_gui)]
        focal = [img_wh[0], img_wh[0]]

        GUI(args, focal, img_wh).render()
        dpg.destroy_context()