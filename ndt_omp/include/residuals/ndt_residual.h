#pragma once

/*****************/
// by ZSN : sophus -> fmt
#define FMT_HEADER_ONLY
#include "fmt/format.h"
/*****************/


#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "sophus/se3.hpp"

class NDTResidual : public ceres::SizedCostFunction<1, 6>
{
  typedef Eigen::Matrix<double, 6, 1> Vector6d;

  public:
    NDTResidual(const Eigen::Vector3d& pts_i, const Eigen::Vector3d& mu, const Eigen::Matrix3d& sigma_inv);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    
    // void check(double **parameters);

    Eigen::Vector3d point_target_, mu_;
    Eigen::Matrix3d sigma_inv_;  // covariance
    // static Eigen::Matrix2d sqrt_info;
    static double sum_t;
    void testNDTresidual();
};


//use auto derivation instead of closed-form Jacobian calculation.

namespace Eigen {
namespace internal {

template <class T, int N, typename NewType>
struct cast_impl<ceres::Jet<T, N>, NewType> {
  EIGEN_DEVICE_FUNC
  static inline NewType run(ceres::Jet<T, N> const &x) {
    return static_cast<NewType>(x.a);
  }
};

} // namespace internal
} // namespace Eigen

class NDTCostFunctor{
public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NDTCostFunctor(const Eigen::Vector3d point_target, const Eigen::Vector3d mu, Eigen::Matrix3d sigma_inv) :point_target_(point_target), mu_(mu), sigma_inv_(sigma_inv){}

  template<typename T>
  bool operator()(const T *const x, T *residual)const{
    // Eigen::Map<Sophus::SE3<T> const> const T_estimated(x);
    Eigen::Map<Eigen::Matrix<T, 6, 1> const> const x_vec(x);
    Sophus::SE3<T> T_estimated = Sophus::SE3<T>::exp(x_vec);
    Eigen::Matrix<T, 3, 1> p_trans =
        T_estimated.rotationMatrix() * point_target_.cast<T>() +
        T_estimated.translation();

    Eigen::Matrix<T, 3, 1> p_delta = p_trans - mu_.cast<T>();

    residual[0] = p_delta.transpose() * sigma_inv_.cast<T>() * p_delta;
    // residual[0] = p_delta[0] + p_delta[1] + p_delta[2];
    // std::cout<<"T_estimated"<<std::endl;
    // std::cout<<T_estimated.matrix()<<std::endl;
    // std::cout<<"point_target"<<std::endl;
    // std::cout<<point_target_<<std::endl;

    return true;
  }

private:
  const Eigen::Vector3d point_target_;
  const Eigen::Vector3d mu_;  // exception of cell
  const Eigen::Matrix3d sigma_inv_;  // covariance matrix of cell
};
