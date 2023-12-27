# metashape2sdfstudio

- 将metashape计算所得的相机位姿文件(pose.xml)转为sdfstudio支持的格式(metadata.json)

- 可将点云(pcd.ply)和mesh(mesh.ply)同步转换到训练场景下
- 添加lookdown场景，针对无人机数据

- 可以对转换前后的位置以及点云等进行可视化
- 暴露关键场景参数：目标中心点位置和缩放因子
- 将总的齐次变换矩阵保存在metadata.json中

- 通过config.ini来读取参数，并在执行代码保存备份

