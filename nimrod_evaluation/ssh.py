
import os
from jumpssh import SSHSession
import glob


WEIGHTS_PATH = r'C:\\HUJI\\lab_weiss\\alias_free_convnets\\alias_free_convnets\\convnext_afc_tiny_f1_gelu_c\\checkpoint-best-ema.pth'
USERNAME = "nimrod.kremer"
PASSWORD = "Nimi2498"
IMAGES_TEST_PATH = "/cs/labs/Academic/dataset/ILSVRC2012/ILSVRC2012_test"
SAVE_PATH = r"C:\HUJI\lab_weiss\alias_free_convnets\images"

class Ssh:
    def __init__(self):
        self.remote_session = self.get_remote_session()
        self.image_names = self.get_img_names

    def get_img_names(self):
        output = self.remote_session.get_cmd_output("ls /cs/labs/Academic/dataset/ILSVRC2012/ILSVRC2012_test/")
        image_names = output.split(" ")
        image_names = [img for img in image_names if img != ""]
        return image_names

    def download_image(self, image_name):
        self.remote_session.get(os.path.join(IMAGES_TEST_PATH, image_name),
                            os.path.join(SAVE_PATH, image_name))

    def delete_files_in_folder(self):
        files = glob.glob(SAVE_PATH)
        for f in files:
            try:
                os.remove(f)
                print(f"Deleted {f}")
            except Exception as e:
                print(f"Failed to delete {f} : {e}")

    def download_images(self, images):
        self.delete_files_in_folder()
        for image in images:
            self.download_image(image)

    def get_remote_session(self):
        otp = input("enter the otp \n")
        gateway_session = SSHSession("bava.cs.huji.ac.il", "nimrod.kremer", password=otp).open()
        remote_session = gateway_session.get_remote_session("river", password=PASSWORD)
        return remote_session