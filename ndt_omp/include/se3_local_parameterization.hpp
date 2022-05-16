#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>

Eigen::Quaterniond atoQuaterniond (const Eigen::Vector3d &v3d){
    Eigen::Quaterniond q;
    q = Eigen::AngleAxisd(v3d[2], Eigen::Vector3d::UnitZ()) * 
        Eigen::AngleAxisd(v3d[1], Eigen::Vector3d::UnitY()) * 
        Eigen::AngleAxisd(v3d[0], Eigen::Vector3d::UnitX());
    return q;
}

Eigen::Vector3d qtoAngleAxis (const Eigen::Quaterniond &q){
    Eigen::Matrix3d rot_mat4 = q.matrix();
    Eigen::Vector3d euler_angle = rot_mat4.eulerAngles(2,1,0);
    double v3[3];
    v3[0] = euler_angle[2];
    v3[1] = euler_angle[1];
    v3[2] = euler_angle[0];
    Eigen::Vector3d v3d(v3);
    return v3d;
}

class LocalParameterizationSE3 : public ceres::LocalParameterization
{
  public:
    virtual ~LocalParameterizationSE3() {}

    // SE3 plus operation for Ceres
    //
    //  exp(delta)*T
    //
    virtual bool Plus(double const *T_raw, double const *delta_raw, double *T_plus_delta_raw) const
    {
        Eigen::Map<Sophus::SE3d const> const T(T_raw);

        Eigen::Matrix<double, 6, 1> T_plus_delta_v6(T_plus_delta_raw);
        Sophus::SE3d T_plus_delta = Sophus::SE3d::exp(T_plus_delta_v6);
        
        // Eigen::Map<Eigen::Vector3d const> delta_rot(delta_raw);
        // Eigen::Map<Eigen::Vector3d const> delta_trans(delta_raw+3);
        // Eigen::Matrix<double,6,1> delta;
        // delta.block<3,1>(0,0) = delta_trans;//translation
        // delta.block<3,1>(3,0) = delta_rot;//rotation
        Eigen::Matrix<double, 6, 1> delta(delta_raw);

        T_plus_delta = Sophus::SE3d::exp(delta) * T;
        double *v7 = T_plus_delta.data();

        Eigen::Quaterniond q(v7[0], v7[1], v7[2], v7[3]);
        T_plus_delta_v6.block<3, 1>(0, 0) = qtoAngleAxis(q);
        Eigen::Map<Eigen::Vector3d const> T_delta_trans(v7+4);
        T_plus_delta_v6.block<3, 1>(3, 0) = T_delta_trans;

        return true;
    }

    virtual bool ComputeJacobian(const double *x, double *jacobian) const
    {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J(jacobian);
        J.setIdentity();
        return true;
    }
    // virtual bool ComputeJacobian(const double* x, double* jacobian) const override;

    // virtual int GlobalSize() const { return Sophus::SE3d::num_parameters; }
    virtual int GlobalSize() const { return 6; }

    virtual int LocalSize() const { return Sophus::SE3d::DoF; }
};

