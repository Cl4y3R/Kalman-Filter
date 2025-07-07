# include <Eigen/Dense>
# include <vector>

class EKF
{
public:
    EKF(const Eigen::VectorXd &x_in, const Eigen::MatrixXd &S_in, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const double &dt);
    ~EKF();

    void predict(Eigen::VectorXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u, const Eigen::VectorXd &param, const double eps);
    void predict(const Eigen::VectorXd &u, const Eigen::MatrixXd &F_);
    void update(Eigen::VectorXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u, const Eigen::VectorXd &param, const double eps, const Eigen::VectorXd &z);
    void update(const Eigen::VectorXd &z, const Eigen::MatrixXd &H_);
    Eigen::MatrixXd numJacobian(Eigen::VectorXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&), const Eigen::VectorXd &x, const Eigen::VectorXd &u, const Eigen::VectorXd &param, const double &h);
    Eigen::VectorXd getEstimation();
    Eigen::MatrixXd qrFactor(const Eigen::MatrixXd &A, const Eigen::MatrixXd &S, const Eigen::MatrixXd &Ns);

private:

    // init flag
    bool init_flag_ = false;

    // sample time
    double dt_;

    // state vector
    Eigen::VectorXd x_;

    // state change vector
    Eigen::VectorXd dx_;

    // upper triangular state covariance matrix
    Eigen::MatrixXd S_;

    // state transition matrix
    Eigen::MatrixXd F_;

    // process covariance matrix
    Eigen::MatrixXd Rq_;

    // measurement matrix
    Eigen::MatrixXd H_;

    // measurement covariance matrix
    Eigen::MatrixXd Rr_;
};


EKF::EKF(const Eigen::VectorXd &x_in, const Eigen::MatrixXd &S_in, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const double &dt){
    dt_ = dt;
    x_ = x_in;
    dx_ = Eigen::VectorXd::Zero(x_in.size());
    S_ = S_in;
    Rq_ = Q.llt().matrixL();
    Rr_ = R.llt().matrixL();
    std::cout<<"Square root of Q(0,0): "<<Rq_(0,0)<<std::endl;
    std::cout<<"Square root of Q(1,1): "<<Rq_(1,1)<<std::endl;
    std::cout<<"Square root of R(0,0): "<<Rr_(0,0)<<std::endl;
    std::cout<<"Square root of R(1,1): "<<Rr_(1,1)<<std::endl;
    std::cout<<"EKF initialized"<<std::endl;
}

EKF::~EKF(){
    std::cout<<"EKF destructed"<<std::endl;
}

void EKF::predict(Eigen::VectorXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u, const Eigen::VectorXd &param, const double eps){
    // numerical jacobian method
    if(init_flag_){
        F_ = numJacobian(func, x_, u, param, eps);
        dx_ = func(x_, u, param);
        x_ = x_ + dx_ * dt_;
        S_ = qrFactor(F_,S_,Rq_);
    }
    else{
        init_flag_ = true;
    }
}

void EKF::predict(const Eigen::VectorXd &u, const Eigen::MatrixXd &F_in){
    // pre-calculated explicit jacobian
    if(init_flag_){
        F_ = F_in;
        dx_ = F_ * x_;
        x_ = x_ + dx_ * dt_;
        S_ = qrFactor(F_,S_,Rq_);
    }
    else{
        init_flag_ = true;
    }
}

void EKF::update(Eigen::VectorXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u, const Eigen::VectorXd &param, const double eps, const Eigen::VectorXd &z){
    // numerical jacobian method
    H_ = numJacobian(func, x_, u, param, eps);
    Eigen::MatrixXd Sy = qrFactor(H_,S_,Rr_);
    Eigen::MatrixXd P = S_* S_.transpose();
    Eigen::MatrixXd Pxy = P * H_.transpose();
    Eigen::MatrixXd K = Pxy * ((Sy * Sy.transpose()).inverse());
    Eigen::VectorXd y = z - func(x_, u, param);
    x_ = x_ + K * y;
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K * H_;
    S_ = qrFactor(A,S_,K*Rr_);
}

void EKF::update(const Eigen::VectorXd &z, const Eigen::MatrixXd &H_in){
    // pre-calculated explicit jacobian
    H_ = H_in;
    Eigen::MatrixXd Sy = qrFactor(H_,S_,Rr_);
    Eigen::MatrixXd P = S_* S_.transpose();
    Eigen::MatrixXd Pxy = P * H_.transpose();
    Eigen::MatrixXd K = Pxy * ((Sy * Sy.transpose()).inverse());
    Eigen::VectorXd y = z - H_ * x_;
    x_ = x_ + K * y;
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K * H_;
    S_ = qrFactor(A,S_,K*Rr_);
}

Eigen::MatrixXd EKF::numJacobian(Eigen::VectorXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&), const Eigen::VectorXd &x, const Eigen::VectorXd &u, const Eigen::VectorXd &param, const double &h){
    Eigen::VectorXd Z = func(x, u, param);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(Z.size(), x.size());
    for(int i = 0; i < Z.size(); i++){
        Eigen::VectorXd x_plus_h = x;
        double eps = std::max(h, h*std::fabs(x_plus_h(i)));
        x_plus_h(i) += eps;
        Eigen::VectorXd Z_plus = func(x_plus_h, u, param);
        J.col(i) = (Z_plus - Z) / eps;
    }

    return J;
}

Eigen::MatrixXd EKF::qrFactor(const Eigen::MatrixXd &A, const Eigen::MatrixXd &S, const Eigen::MatrixXd &Ns){
    Eigen::MatrixXd M(S.cols() + Ns.cols(), A.cols());
    M.topRows(S.cols()) = S.transpose() * A.transpose();
    M.bottomRows(Ns.cols()) = Ns.transpose();

    // economy mode, only calculate R
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    Eigen::MatrixXd R = qr.matrixQR()
                         .topRows(std::min(M.rows(), M.cols()))
                         .triangularView<Eigen::Upper>();

    return R.transpose();
}

Eigen::VectorXd EKF::getEstimation(){
    return x_;
}
