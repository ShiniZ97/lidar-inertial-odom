#include "residuals/ndt_residual.h"

NDTResidual::NDTResidual(const Eigen::Vector3d& point_target, const Eigen::Vector3d& mu, const Eigen::Matrix3d& sigma_inv) : point_target_(point_target), mu_(mu), sigma_inv_(sigma_inv){
}

// when ceres::AddResidual, execu
bool NDTResidual::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const{
    // Eigen::Map<Sophus::SE3d const> const T_estimated(*parameters);
    Eigen::Map<Eigen::Matrix<double, 6, 1> const>const vecX(*parameters);
    Sophus::SE3d T_estimated = Sophus::SE3d::exp(vecX);
    Eigen::Map<Eigen::Vector3d> residual_vec(residuals);

    Eigen::Vector3d p_trans = T_estimated.rotationMatrix() * point_target_  + T_estimated.translation();
    Eigen::Vector3d p_delta = p_trans - mu_;

    // Eigen::Matrix3d sqrt_sigma_inv = Eigen::LLT<Eigen::Matrix3d>(sigma_inv_).matrixL().transpose();


    // residual_vec = sqrt_sigma_inv * p_delta;
    residuals[0] = p_delta.transpose() * sigma_inv_ * p_delta;

    // if(jacobians){
    //   Eigen::Map<Eigen::Matrix<double, 1, 6>> jacVec(jacobians[0]);
    //   Eigen::Matrix<double, 3, 6> J_Tp_epsilon;
    //   J_Tp_epsilon.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    //   J_Tp_epsilon.block<3, 3>(0, 3) = -Sophus::SO3d::hat(p_trans);
    //   jacVec = 2 * p_delta.transpose() * sigma_inv_ * J_Tp_epsilon;
    // }
    if(jacobians){
      Eigen::Map<Eigen::Matrix<double, 1, 6>> jacVec(jacobians[0]);
      Eigen::Matrix<double, 3, 6> J_Tp_epsilon;
      J_Tp_epsilon.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      J_Tp_epsilon.block<3, 3>(0, 3) = -Sophus::SO3d::hat(p_trans);
      jacVec = 2 * p_delta.transpose() * sigma_inv_ * J_Tp_epsilon;
    }
    // std::cout<<"the residual is: "<<std::endl;
    // std::cout<<residual_vec.transpose()<<std::endl;
    // std::cout<<"the distance is: "<<std::endl;
    // std::cout<<p_delta.transpose()<<std::endl;

    // if(jacobians){
    //   Eigen::Map<Eigen::Matrix<double, 3, 6>> jacVec(jacobians[0]);
    //   Eigen::Matrix<double, 3, 6> J_Tp_epsilon;
    //   J_Tp_epsilon.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    //   J_Tp_epsilon.block<3, 3>(0, 3) = -Sophus::SO3d::hat(p_trans);
    //   jacVec = sqrt_sigma_inv * J_Tp_epsilon;
    // }

    return true;
 }
