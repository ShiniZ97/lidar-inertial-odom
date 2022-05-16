/*****************/
// by ZSN : sophus -> fmt
#define FMT_HEADER_ONLY
#include "fmt/format.h"
/*****************/
#include "se3_local_parameterization.hpp"

#include "ndt_lm.h"
#include "residuals/ndt_residual.h"
#include "sophus/se3.hpp"
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
n *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_REGISTRATION_NDT_LM_IMPL_H_
#define PCL_REGISTRATION_NDT_LM_IMPL_H_

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget>
omatcher::NormalDistributionsTransform<PointSource, PointTarget>::NormalDistributionsTransform ()
  : target_cells_ ()
  , resolution_ (1.0f)
  , step_size_ (0.1)
  , outlier_ratio_ (0.55)
  , trans_probability_ ()
{
  reg_name_ = "NormalDistributionsTransform";

  transformation_epsilon_ = 0.1;
  max_iterations_ = 35;

  search_method = DIRECT7;
  num_threads_ = omp_get_max_threads();
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget> void
omatcher::NormalDistributionsTransform<PointSource, PointTarget>::computeTransformation (PointCloudSource &output, const Eigen::Matrix4f &guess)
{
  // if (guess != Eigen::Matrix4f::Identity ())
  // {
  //   // Initialise final transformation to the guessed one
  //   final_transformation_ = guess;
  //   // Apply guessed transformation prior to search for neighbours
  //   transformPointCloud (output, output, guess);
  // }
  Sophus::SE3d guess_SE3(guess.cast<double>());
  // std::cout<<"guess_SE3: "<<guess_SE3.rotationMatrix()<<std::endl;
  double* pose_result = guess_SE3.data(); // 7D, 4 (q) + 3 (t_xyz)

  // ******************************
  Eigen::Quaterniond q_(pose_result[0], pose_result[1], pose_result[2], pose_result[3]);
  Eigen::Vector3d v3d = qtoAngleAxis(q_);
  double pose_result_[6] = {v3d[0], v3d[1], v3d[2], pose_result[4], pose_result[5], pose_result[6]};
  // ******************************

  // Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor>ae eig_transformation;
  // eig_transformation.matrix () = final_transformation_;

  for(int itr_num = 0; itr_num < 5; itr_num++){
    // Eigen::Matrix<double, 6, 1> p_cur(pose_result);

    // ***************************
    Eigen::Matrix<double, 6, 1> p_cur(pose_result_);
    // ***************************

    Sophus::SE3d pose_cur_iteration = Sophus::SE3d::exp(p_cur);
    ceres::Problem ndt_problem;
    ceres::LossFunction *loss_fuc;

    // ndt_problem.AddParameterBlock(pose_result, 6);  // FIXEME: hard_coded block_size

    // ****************************
    ceres::LocalParameterization *q_quaternion = new LocalParameterizationSE3();
    ndt_problem.AddParameterBlock(pose_result_, 6, q_quaternion);  // FIXEME: hard_coded block_size
    // ****************************

    loss_fuc = new ceres::CauchyLoss(1.0);
    // Convert initial guess matrix to 6 element transformation vector
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distance;
    PointCloudSource updated_pointCloud;
    transformPointCloud (output, updated_pointCloud, pose_cur_iteration.matrix());

    // TODO: openmp speed up
    for (int idx = 0; idx < output.size(); idx++) {
      PointSource x_trans_pt = output.points[idx];
      PointSource x_trans_quary = updated_pointCloud.points[idx];
      switch(search_method){
      case KDTREE:
        target_cells_.radiusSearch(x_trans_quary, resolution_, neighborhood, distance);
        break;
      case DIRECT26:
        target_cells_.getNeighborhoodAtPoint(x_trans_quary, neighborhood);
        break;
      default:
      case DIRECT7:
        target_cells_.getNeighborhoodAtPoint7(x_trans_quary, neighborhood);
        break;
      case DIRECT1:
        target_cells_.getNeighborhoodAtPoint1(x_trans_quary, neighborhood);
        break;
      }

      Eigen::Vector3d point_vec;
      point_vec << x_trans_pt.x, x_trans_pt.y, x_trans_pt.z;


      // closed-formed Jacobin calculation
      for (auto ref_iter = neighborhood.begin(); ref_iter != neighborhood.end(); ++ref_iter){
        TargetGridLeafConstPtr cell = *ref_iter;

        NDTResidual *res = new NDTResidual(point_vec, cell->getMean(), cell->getInverseCov());
        ndt_problem.AddResidualBlock( res, loss_fuc, pose_result);
      }

      // auto-derivation test
      
      // for (auto ref_iter = neighborhood.begin(); ref_iter != neighborhood.end(); ++ref_iter){
      //   TargetGridLeafConstPtr cell = *ref_iter;

      //   ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<NDTCostFunctor, 1, 6>( new NDTCostFunctor( point_vec, cell->getMean(), cell->getInverseCov()));
      //   // Eigen::Map<Sophus::SE3d const> const T_estimated_2(pose_result);
      //   // std::cout << "T_estimated: " << T_estimated_2.matrix() << std::endl;
      //   ndt_problem.AddResidualBlock( cost_function, loss_fuc, pose_result);
      // }
    }
    bool shallVerbose = 0; 
    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
    solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.max_num_iterations = 5;
    // solver_options.num_threads = num_threads_;
    solver_options.num_threads = 1;

    ceres::Solver::Summary summary;

    ceres::Solve(solver_options, &ndt_problem, &summary);
    std::cout<<summary.FullReport()<<std::endl;

    std::cout<<"final result: "<<std::endl;
    Eigen::Matrix<double, 6, 1> pVec(pose_result);
    Sophus::SE3d finalResult = Sophus::SE3d::exp(pVec);

    // Sophus::SE3d()
    std::cout<<finalResult.matrix()<<std::endl;
  }
  // FIXME: setting function



}

#ifndef _OPENMP
int omp_get_max_threads() { return 1; }
int omp_get_thread_num() { return 0; }
#endif

#endif // PCL_REGISTRATION_NDT_IMPL_H_
