# File      :metashpe_to_sdfstudio.py
# Auther    :WooChi
# Time      :2023/03/21
# Version   :1.0
# Function  :

from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
from icecream import ic
import world_to_scene as pt
import argparse
import xml.etree.ElementTree as ET
import math
import cv2
import numpy as np
import json
import open3d as o3d
import os
import shutil


class Metashape2Sdfstudio():

    def __int__(self):
        self.fl_x = 0
        self.fl_y = 0
        self.k1 = 0
        self.k2 = 0
        self.p1 = 0
        self.p2 =0
        self.cx = 0
        self.cy = 0
        self.camera_angle_x = 0
        self.camera_angle_y = 0
        self.poses=[]
        self.frames=[]
        # world to scene  将世界坐标系转为自定义场景坐标系
        self.w2s=[]   
        self.meta_data=[]

    def get_w2s(self):
        return self.w2s

    def get_meta_data(self):
        return self.meta_data

    def closest_point_2_lines(self,oa, da, ob,
                              db):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
        da = da / np.linalg.norm(da)
        db = db / np.linalg.norm(db)
        c = np.cross(da, db)
        denom = np.linalg.norm(c) ** 2
        t = ob - oa
        ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
        tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
        if ta > 0:
            ta = 0
        if tb > 0:
            tb = 0
        return (oa + ta * da + ob + tb * db) * 0.5, denom

    def central_point(self,out):
        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = np.array(f["transform_matrix"])[0:3, :]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3, :]
                p, w = self.closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                if w > 0.0001:
                    totp += p * w
                    totw += w
        totp /= totw
        print(totp)  # the cameras are looking at totp
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] -= totp
            f["transform_matrix"] = f["transform_matrix"].tolist()
        return out




    def get_calibration(self,root):
        for sensor in root[0][0]:
            for child in sensor:
                if child.tag == "calibration":
                    return child
        print("No calibration found")
        return None

    def reflectZ(self):
        return [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]]

    def reflectY(self):
        return [[1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]

    def matrixMultiply(self,mat1, mat2):
        return np.array([[sum(a * b for a, b in zip(row, col)) for col in zip(*mat2)] for row in mat1])

    def read_metashape_xml(self, args):
        self.args=args
        print(self.args)
        XML_LOCATION = args.xml_path
        IMGTYPE = args.img_type


        with open(XML_LOCATION, "r") as f:
            root = ET.parse(f).getroot()
            # print(root[0][0][0].tag)
            w = float(root[0][0][0][0].get("width"))
            h = float(root[0][0][0][0].get("height"))

            calibration = self.get_calibration(root)
            self.fl_x = float(calibration[1].text)
            self.fl_y = self.fl_x
            self.k1 = float(calibration[4].text)
            self.k2 = float(calibration[5].text)
            self.p1 = float(calibration[7].text)
            self.p2 = float(calibration[8].text)
            self.cx = float(calibration[2].text) + w / 2
            self.cy = float(calibration[3].text) + h / 2
            self.camera_angle_x = math.atan(float(w) / (float(self.fl_x) * 2)) * 2
            self.camera_angle_y = math.atan(float(h) / (float(self.fl_y) * 2)) * 2


            self.frames = list()
            self.poses=[]
            for frame in root[0][2]:
                current_frame = dict()
                if not len(frame):
                    continue
                if (frame[0].tag != "transform"):
                    continue

                imagePath =  frame.get("label") + "." + IMGTYPE
                current_frame.update({"file_path": imagePath})
                matrix_elements = [float(i) for i in frame[0].text.split()]
                transform_matrix = np.array(
                    [[matrix_elements[0], matrix_elements[1], matrix_elements[2], matrix_elements[3]],
                     [matrix_elements[4], matrix_elements[5], matrix_elements[6], matrix_elements[7]],
                     [matrix_elements[8], matrix_elements[9], matrix_elements[10], matrix_elements[11]],
                     [matrix_elements[12], matrix_elements[13], matrix_elements[14], matrix_elements[15]]])

                transform_matrix = transform_matrix[[0, 1,2, 3], :]
                current_frame.update(
                    {"transform_matrix": transform_matrix})
                # ic(transform_matrix )
                # ic(current_frame)
                self.poses.append(transform_matrix)
                self.frames.append(current_frame)

        # ic(self.poses)
        # ic(self.frames)


    def write_sdfstudio_json(self,move_pic=True):
        """
        given data that follows the nerfstduio format such as the output from colmap or polycam,
        convert to a format that sdfstudio will ingest
        """
        args=self.args
        output_dir = Path(self.args.output_dir)
        input_dir = Path(self.args.img_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # === load camera intrinsics and poses ===
        cam_intrinsics = []
        cam_intrinsics.append(np.array([
            [self.fl_x, 0, self.cx],
            [0, self.fl_y, self.cy],
            [0, 0, 1]]))

        image_paths = []
        # only load images with corresponding pose info
        # currently in random order??, probably need to sort
        for frame in self.frames:
        # load intrinsics from polycam

        # load poses
        # OpenGL/Blender convention, needs to change to COLMAP/OpenCV convention
        # https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
        # IGNORED for now

            # load images
            file_path = Path(frame["file_path"])
            img_path = input_dir / file_path.name
            assert img_path.exists()
            image_paths.append(img_path)

        poses=self.poses


        # Check correctness
        assert len(poses) == len(image_paths)
        assert len(poses) == len(cam_intrinsics) or len(cam_intrinsics) == 1

        # Filter invalid poses
        poses = np.array(poses)
        valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
        # plot_linear_cube(poses)
        # === Normalize the scene ===

        if self.args.scene_type in ["indoor", "object"]:
            poses_new, scale_mat = pt.scene_true(poses)

        elif self.args.scene_type in ["lookdown"]:
            poses_new, scale_mat = pt.scene_lookdown(poses, np.array(self.args.center_target),self.args.scale_value)

        elif self.args.scene_type in ["lookdown_unbound"]:
            poses_new, scale_mat = pt.scene_lookdown(poses, np.array(self.args.center_target),-1 )

        self.w2s=scale_mat
        self.poses=poses_new
        # === Construct the scene box ===
        if args.scene_type == "indoor":
            scene_box = {
                "aabb": [[-1, -1, -1], [1, 1, 1]],
                "near": 0.05,
                "far": 2.5,
                "radius": 1.0,
                "collider_type": "box",
            }
        elif args.scene_type == "object":
            scene_box = {
                "aabb": [[-1, -1, -1], [1, 1, 1]],
                "near": 0.05,
                "far": 2.0,
                "radius": 1.0,
                "collider_type": "near_far",
            }

        elif self.args.scene_type == "lookdown":
            scene_box = {
                "aabb": [[-1, -1, -1], [1, 1, 1]],
                "near": 0.05,
                "far": 2.0,
                "radius": 1.0,
                "collider_type": "near_far",
            }
        elif self.args.scene_type == "lookdown_unbound":
            p_new = poses_new[:, 0:3, 3]
            F_H = args.center_target[2]  # flight_height
            scene_box = {
                "aabb": [[np.min(p_new, axis=0)[0], np.min(p_new, axis=0)[1], -0.2 * F_H],
                         [np.max(p_new, axis=0)[0], np.max(p_new, axis=0)[1], 0.5 * F_H]],
                "near": 0.5 * F_H,
                "far": 1.2 * F_H,
                "radius": 0.6 * F_H,
                "collider_type": "box",
            }
            self.poses= poses_new



        # === Resize the images and intrinsics ===
        # Only resize the images when we want to use mono prior
        sample_img = cv2.imread(str(image_paths[0]))
        h, w, _ = sample_img.shape

        tar_h, tar_w = h, w

        # === Construct the frames in the meta_data.json ===
        frames = []
        out_index = 0
        for idx, (valid, pose, image_path) in enumerate(tqdm(zip(valid_poses, self.poses, image_paths))):
            if not valid:
                continue
            out_img_path = output_dir / os.path.basename(image_path)
            shutil.copy(image_path, out_img_path )

            if move_pic:
                shutil.copy(image_path, out_img_path )

            rgb_path = str(out_img_path.relative_to(output_dir))

            frame = {
                "rgb_path": rgb_path,
                "camtoworld": pose.tolist(),
                "intrinsics": cam_intrinsics[0].tolist()
            }

            frames.append(frame)
            out_index += 1

        # 将7参数转为齐次变换矩阵
        R = scale_mat[:3, :3]
        T = scale_mat[:3, 3]
        s = scale_mat[3, 3]

        # 创建齐次变换矩阵
        M = np.zeros((4, 4))
        M[:3, :3] = s * R
        M[:3, 3] = T
        M[3, 3] = 1
        self.w2s=M
        M_inv = np.linalg.inv(M)
        print("齐次变换矩阵:")
        print(M)
        print("齐次变换矩阵:")
        print(M_inv)


        # === Construct and export the metadata ===
        meta_data = {
            "camera_model": "OPENCV",
            "height": tar_h,
            "width": tar_w,
            "has_pointclouds":True,
            "pcd_path":"pcd.txt",
            "has_mono_prior": False,
            "has_sensor_depth": False,
            "has_foreground_mask": False,
            "pairs": None,
            "w2c": M.tolist(),
            "w2c_inv": M_inv.tolist(),
            "scene_box": scene_box,
            "frames": frames,
        }
        self.meta_data=meta_data
        with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4)

        print(f"Done! The processed data has been saved in {output_dir}")


    def transform_pts(self,pts):
        ic(self.w2s)


        # pts_new = np.dot(self.w2s[:3, :3], np.transpose(pts_xyz))
        # pts_new *= self.w2s[3, 3]
        # pts_new = pts_new + np.array([self.w2s[:3, 3]]).T
        # pts_all=np.concatenate((pts_new.T,pts[:,3:]), axis=1)

        # 将点云转换为齐次坐标
        homogeneous_points = np.hstack((pts, np.ones((pts.shape[0], 1))))
        # 对点云进行变换
        transformed_points = np.dot(homogeneous_points, self.w2s.T)
        # 去除齐次坐标的最后一列，得到变换后的点云
        transformed_points = transformed_points[:, :3]
        return transformed_points



    def transform_and_save_ply(self,input_ply_path, output_ply_path):

        mesh = o3d.io.read_triangle_mesh(input_ply_path)

        # 提取顶点数据
        vertices = np.asarray(mesh.vertices)
        # 对顶点进行变换
        pts=self.transform_pts(vertices)
        # 替换回原来的顶点
        mesh.vertices = o3d.utility.Vector3dVector(pts)
        # 保存为PLY文件
        o3d.io.write_triangle_mesh(output_ply_path, mesh)



def parse_args():
    parser = argparse.ArgumentParser(
        description="preprocess metashape camera pose(.xml) to sdfstudio camera pose(.json)")
    parser.add_argument("--xml-path", dest="xml_path", required=True, help="specify xml file location")
    parser.add_argument("--img-folder", dest="img_folder", required=True, help="location of image folder")
    parser.add_argument("--img-type", dest="img_type", required=False, help="type of images (ex. jpg, png, ...)",
                        default="JPG")
    parser.add_argument("--output-dir", dest="output_dir", required=False, help="path to ouput data directory",
                        default="transforms.json")
    parser.add_argument("--scene-type", dest="scene_type", required=True,
                        choices=[ "lookdown", "lookdown_unbound","object","indoor"],default="lookdown",
                        help="The scene will be normalized into a unit sphere when selecting indoor or object.")
    parser.add_argument("--center-target", dest="center_target", type=float, nargs=3,
                        default=[0, 0, 1],
                        help="center_target")
    parser.add_argument("--scale_value", dest="scale_value", required=False,type=float, help="scale value",
                        default=2)
    args = parser.parse_args()
    return args




if __name__ == "__main__":

    # --xml T:\ProjectData\MetaShape_Project\cameras4sdfstudio_lab04.xml --img-folder D:\Project\CmakeProject\instant-ngp\data\nerf\UAV_01\images\ --img-type JPG --scene-type lookdown --output-dir uav
    # --xml T:\ProjectData\MetaShape_Project\cameras4sdfstudio_lab04.xml --img-folder D:\Project\CmakeProject\instant-ngp\data\nerf\UAV_01\images\ --img-type JPG --scene-type uav --output-dir uav --center-target 0 0 7
    # --xml T:\ProjectData\MetaShape_Project\cameras4sdfstudio_earphone.xml --img-folder D:\Project\CmakeProject\instant-ngp\data\nerf\earphone_2\images\ --img-type JPG --scene-type uav --output-dir earphone --center-target 0 0 2
    # --xml T:\ProjectData\MetaShape_Project\lab_selected.xml --img-folder T:\ProjectData\MetaShape_Project\labrary_selected\ --img-type JPG --scene-type lookdown --output-dir lab

    # --xml T:\ProjectData\MetaShape_Project\select_uav_image\small_area_poses.xml --img-folder T:\ProjectData\MetaShape_Project\select_uav_image\small_area\ --img-type JPG --scene-type lookdown --output-dir small_area
    # --xml T:\ProjectData\MetaShape_Project\courthouse_5.xml --img-folder E:\Courthouse\courthouse_sampled_5\ --img-type JPG --scene-type object --output-dir courthouse_5
    # --xml E:\Courthouse\courhouse_front.xml --img-folder E:\Courthouse\courthouse\ --img-type JPG --scene-type object --output-dir courthouse_front
    # --xml E:\Courthouse\front_rsz.xml --img-folder E:\Courthouse\courhouse_front\resize\ --img-type JPG --scene-type object --output-dir front_rsz

    #--xml E:\Courthouse\front_rsz.xml --img-folder E:\Courthouse\courhouse_front\resize\ --img-type JPG --scene-type object --output-dir front_rsz
    # 0629 courthouse
    # --xml data\courthouse_pose.xml --img-folder E:\Courthouse\courthouse\ --img-type JPG --scene-type object --output-dir courthouse

    # --xml data\courthouse_front.xml --img-folder E:\Courthouse\courthouse\ --img-type JPG --scene-type object --output-dir courthouse_front
    # --xml data\csu_gate.xml --img-folder D:\Project\PythonProject\Extract_picture_from_video\pic_xm\ --img-type JPG --scene-type object --output-dir csu_gate

    # 0714
    # --xml T:\ProjectData\MetaShape_Project\cameras4sdfstudio_lab04.xml --img-folder D:\Project\CmakeProject\instant-ngp\data\nerf\UAV_01\images\ --img-type JPG --scene-type uav --output-dir uav --center-target 0 0 7

    # 0801
    # --xml data/Drone/pose.xml --img-folder T:\ProjectData\Undistorted_images_full_res\resize\ --img-type JPG --scene-type lookdown --output-dir drone
    # 0803
    # --xml data/rathaus/pose.xml --img-folder T:\ProjectData\uav_data\rathaus\resize\sample\ --img-type JPG --scene-type lookdown --output-dir rathaus


    # --xml data/rathaus2/pose.xml --img-folder T:\ProjectData\uav_data\rathaus\resize\sample_6\ --img-type JPG --scene-type lookdown --output-dir rathaus2


    # pts_path = r"data/sea_0823/pcd_sub.txt"
    #
    # pts = np.loadtxt(pts_path, delimiter=" ")
    # pts_center=np.mean(pts[:,:3],axis=0)
    # ic(pts_center)
    args = parse_args()
    a=Metashape2Sdfstudio()
    a.read_metashape_xml(args)
    a.write_sdfstudio_json(move_pic=False)
    # pts = a.transform_pts(pts)

    ic(args.output_dir)
    # np.savetxt(args.output_dir+"/pcd.txt",pts,delimiter=" ",fmt='%.3f')

    # ply_path = r"data/sea_0823/mesh.ply"
    # a.transform_and_save_ply(ply_path,args.output_dir+"/mesh.ply")

    # pt.plot_linear_cube(pts=pts)






