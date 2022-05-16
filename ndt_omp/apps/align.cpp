#include <iostream>
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>
#include <pclomp/ndt_lm.h>

Eigen::Matrix4f result_global;

// align point clouds and measure processing time
pcl::PointCloud<pcl::PointXYZ>::Ptr align(pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr registration, const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud ) {
  registration->setInputTarget(target_cloud);
  registration->setInputSource(source_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());

  auto t1 = ros::WallTime::now();
  registration->align(*aligned);
  auto t2 = ros::WallTime::now();
  std::cout << "single : " << (t2 - t1).toSec() * 1000 << "[msec]" << std::endl;

  for(int i=0; i<10; i++) {
    registration->align(*aligned);
  }
  auto t3 = ros::WallTime::now();
  std::cout << "10times: " << (t3 - t2).toSec() * 1000 << "[msec]" << std::endl;
  std::cout << "fitness: " << registration->getFitnessScore() << std::endl << std::endl;

  result_global = registration->getFinalTransformation();
  std::cout << registration->getFinalTransformation()<<std::endl;

  return aligned;
}


int main(int argc, char** argv) {
  if(argc != 3) {
    std::cout << "usage: align target.pcd source.pcd" << std::endl;
    return 0;
  }

  std::string target_pcd = argv[1];
  std::string source_pcd = argv[2];

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());


  if(pcl::io::loadPCDFile(target_pcd, *target_cloud)) {
    std::cerr << "failed to load " << target_pcd << std::endl;
    return 0;
  }
  if(pcl::io::loadPCDFile(source_pcd, *source_cloud)) {
    std::cerr << "failed to load " << source_pcd << std::endl;
    return 0;
  }

  // downsampling
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());

  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

  voxelgrid.setInputCloud(target_cloud);
  voxelgrid.filter(*downsampled);
  *target_cloud = *downsampled;

  voxelgrid.setInputCloud(source_cloud);
  voxelgrid.filter(*downsampled);
  source_cloud = downsampled;

  ros::Time::init();

  // benchmark
  std::cout << "--- pcl::GICP ---" << std::endl;
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr gicp(new pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned = align(gicp, target_cloud, source_cloud);

  // std::cout << "--- pclomp::GICP ---" << std::endl;
  // pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr gicp_omp(new pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>());
  // aligned = align(gicp_omp, target_cloud, source_cloud);

  // std::cout << "--- pcl::NDT ---" << std::endl;
  // pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt(new pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  // ndt->setResolution(1.0);
  // aligned = align(ndt, target_cloud, source_cloud);

  // std::vector<int> num_threads = {1, omp_get_max_threads()};
  std::vector<int> num_threads = {1, 8};

  std::vector<std::pair<std::string, pclomp::NeighborSearchMethod>> search_methods = {
    {"KDTREE", pclomp::KDTREE},
    {"DIRECT7", pclomp::DIRECT7}
    // {"DIRECT1", pclomp::DIRECT1}
  };

  std::cout << "--- pcl::NDT dogleg ---" << std::endl;


  pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp(new pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  ndt_omp->setResolution(1.0);

  for(int n : num_threads) {
    for(const auto& search_method : search_methods) {
      std::cout << "--- pclomp::NDT (" << search_method.first << ", " << n << " threads) ---" << std::endl;
      ndt_omp->setNumThreads(n);
      ndt_omp->setNeighborhoodSearchMethod(search_method.second);
      aligned = align(ndt_omp, target_cloud, source_cloud);
    }
  }

  omatcher::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr
      ndt_lm(new omatcher::NormalDistributionsTransform<pcl::PointXYZ,
                                                        pcl::PointXYZ>());

  ndt_lm->setInputTarget(target_cloud);
  ndt_lm->setInputSource(source_cloud);
  ndt_lm->setResolution(1);
  ndt_lm->setNeighborhoodSearchMethod(omatcher::DIRECT7);
  pcl::PointCloud<pcl::PointXYZ>::Ptr lm_result(
      new pcl::PointCloud<pcl::PointXYZ>());
  ndt_lm->setMaximumIterations(3);
  result_global.block<3, 3>(0, 0) = Eigen::Matrix<float, 3, 3>::Identity();
  std::cout << "intial guess: " << result_global << std::endl;

  auto tt1 = ros::WallTime::now();
  ndt_lm->align(*lm_result, result_global);
  // ndt_lm->align(*lm_result);
  
  auto tt2 = ros::WallTime::now();
  std::cout << "single lm: " << (tt2 - tt1).toSec() * 1000 << "[msec]"
            << std::endl;

  // visulization
  pcl::visualization::PCLVisualizer vis("vis");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_handler(target_cloud, 255.0, 0.0, 0.0); // red : 255
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_handler(source_cloud, 0.0, 255.0, 0.0); // green : 255
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_handler(aligned, 0.0, 0.0, 255.0); // blue : 255
  vis.addPointCloud(target_cloud, target_handler, "target");
  vis.setBackgroundColor(255, 255, 255);
  vis.addPointCloud(source_cloud, source_handler, "source");
  vis.addPointCloud(aligned, aligned_handler, "aligned");
  vis.spin();

  // // save pcd
  // pcl::io::savePCDFileASCII<pcl::PointXYZ>("/home/sn/lp_aligned.pcd", *aligned);

  return 0;
}
 
