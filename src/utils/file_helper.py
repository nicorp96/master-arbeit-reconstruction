import os


class FileHelper:
    BASE_PATH = os.path.abspath(os.getcwd())

    def get_object_path(
        self,
        folder_base: str,
        folder_sub_base: str,
        object_folder_name: str,
        object_name: str,
        object_type: str,
    ):
        path = os.path.join(
            self.BASE_PATH, folder_base, object_folder_name, folder_sub_base
        )
        self.check_if_path_exist(path)
        if path is not None:
            return os.path.join(path, object_name + object_type)
        return None

    def get_parent_of_object_path(
        self, folder_base: str, folder_sub_base: str, object_name: str
    ):
        path = os.path.join(self.BASE_PATH, folder_base, object_name, folder_sub_base)
        self.check_if_path_exist(path=path)
        return path

    def get_trajectory_path(
        self,
        base_folder: str,
        object_folder_name: str,
        folder_sub_base: str,
        trj_name: str,
    ):
        path = os.path.join(
            self.BASE_PATH, base_folder, object_folder_name, folder_sub_base
        )
        self.check_if_path_exist(path=path)
        path = os.path.join(
            path,
            trj_name,
        )
        return path

    def get_path_camera_info(self, base_folder, folder_name,intrinsic_name):
        path = os.path.join(
            self.BASE_PATH, base_folder, folder_name
        )
        self.check_if_path_exist(path=path)
        path = os.path.join(
            path,
            intrinsic_name,
        )
        return path

    def get_path_calibration(self, base_folder, folder_name):
        path = os.path.join(
            self.BASE_PATH, base_folder, folder_name
        )
        self.check_if_path_exist(path=path)
        return path

    def check_if_path_exist(self, path):
        is_exist = os.path.exists(path)
        if not is_exist:
            os.makedirs(path)

    def remove_file_in_dir(self, file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)

    def remove_files_in_dir(self, path: str):
        if os.path.exists(path):
            for file in os.scandir(path):
                os.remove(file.path)
