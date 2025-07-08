# include <Eigen/Dense>
# include <Eigen/Core>
# include <vector>

class EKF
{
public:
    EKF(const Eigen::VectorXd &x_in, const Eigen::MatrixXd &S_in, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const double &dt, const bool &num_opt_flag = false);
    ~EKF();

    void predict(Eigen::VectorXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u, const Eigen::VectorXd &param, const double eps);
    void predict(const Eigen::VectorXd &u, const Eigen::MatrixXd &F_);
    void update(Eigen::VectorXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u, const Eigen::VectorXd &param, const double eps, const Eigen::VectorXd &z);
    void update(const Eigen::VectorXd &z, const Eigen::MatrixXd &H_);
    void setJacobian(const Eigen::MatrixXd &F_in, const Eigen::MatrixXd &H_in);
    Eigen::MatrixXd numJacobian(Eigen::VectorXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&), const Eigen::VectorXd &x, const Eigen::VectorXd &u, const Eigen::VectorXd &param, const double &h);
    Eigen::VectorXd getEstimation();
    Eigen::MatrixXd qrFactor(const Eigen::MatrixXd &A, const Eigen::MatrixXd &S, const Eigen::MatrixXd &Ns);

private:
    // numerical optimization flag
    bool num_opt_flag_;

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


EKF::EKF(const Eigen::VectorXd &x_in, const Eigen::MatrixXd &S_in, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const double &dt, const bool &num_opt_flag)
        :x_(x_in), dt_(dt), S_(S_in), Rq_(Q), Rr_(R), num_opt_flag_(num_opt_flag)
{
    dx_ = Eigen::VectorXd::Zero(x_in.size());
    if (num_opt_flag_ == true){
        Rq_ = Q.llt().matrixL();
        Rr_ = R.llt().matrixL();
        std::cout<<"Square root of Q(0,0): "<<Rq_(0,0)<<std::endl;
        std::cout<<"Square root of Q(1,1): "<<Rq_(1,1)<<std::endl;
        std::cout<<"Square root of R(0,0): "<<Rr_(0,0)<<std::endl;
        std::cout<<"Square root of R(1,1): "<<Rr_(1,1)<<std::endl;
    }
    else{
        Rq_ = Q;
        Rr_ = R;
        std::cout<<"Q(0,0): "<<Rq_(0,0)<<std::endl;
        std::cout<<"Q(1,1): "<<Rq_(1,1)<<std::endl;
        std::cout<<"R(0,0): "<<Rr_(0,0)<<std::endl;
        std::cout<<"R(1,1): "<<Rr_(1,1)<<std::endl;
    }
    std::cout<<"EKF initialized"<<std::endl;
}

EKF::~EKF(){
    std::cout<<"EKF destructed"<<std::endl;
}

void EKF::predict(Eigen::VectorXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u, const Eigen::VectorXd &param, const double eps){
    // numerical jacobian method, jacobian matrix calculated externally every step
    if(init_flag_){
        dx_ = func(x_, u, param);
        x_ = x_ + dx_ * dt_;
        if (num_opt_flag_ == true){
            S_ = qrFactor(F_, S_, Rq_);
        }
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
        if (num_opt_flag_ == true){
            S_ = qrFactor(F_, S_, Rq_);
        }
    }
    else{
        init_flag_ = true;
    }
}

void EKF::update(Eigen::VectorXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u, const Eigen::VectorXd &param, const double eps, const Eigen::VectorXd &z){
    // numerical jacobian method, jacobian matrix calculated externally every step
    Eigen::MatrixXd K;
    Eigen::MatrixXd P;
    if (num_opt_flag_ == true){
        Eigen::MatrixXd Sy = qrFactor(H_, S_, Rr_);
        P = S_* S_.transpose();
        Eigen::MatrixXd Pxy = P * H_.transpose();
        K = Pxy * ((Sy * Sy.transpose()).inverse());
    }
    else{
        P = F_ * S_ * F_.transpose() + Rq_;
        K = P * H_.transpose() * ((H_ * P * H_.transpose() + Rr_).inverse());
    }
    
    Eigen::VectorXd y = func(x_, u, param);
    x_ = x_ + K * (z - y);
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K * H_;
    if (num_opt_flag_ == true){
        S_ = qrFactor(A, S_, K * Rr_);
    }
    else{
        S_ = A * P;
    }   
}

void EKF::update(const Eigen::VectorXd &z, const Eigen::MatrixXd &H_in){
    // pre-calculated explicit jacobian
    H_ = H_in;
    Eigen::MatrixXd K;
    Eigen::MatrixXd P;
    if (num_opt_flag_ == true){
        Eigen::MatrixXd Sy = qrFactor(H_, S_, Rr_);
        P = S_* S_.transpose();
        Eigen::MatrixXd Pxy = P * H_.transpose();
        K = Pxy * ((Sy * Sy.transpose()).inverse());
    }
    else{
        P = F_ * S_ * F_.transpose() + Rq_;
        K = P * H_.transpose() * ((H_ * P * H_.transpose() + Rr_).inverse());
    }
    
    Eigen::VectorXd y = H_ * x_;
    x_ = x_ + K * (z - y);
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K * H_;
    if (num_opt_flag_ == true){
        S_ = qrFactor(A, S_, K * Rr_);
    }
    else{
        S_ = P;
    }   
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
    const int  min_dim = std::min(M.rows(), M.cols());
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    Eigen::MatrixXd R = qr.matrixQR()
                         .topRows(min_dim)
                         .triangularView<Eigen::Upper>();

    return R.transpose();
}

void EKF::setJacobian(const Eigen::MatrixXd &F_in, const Eigen::MatrixXd &H_in){
    F_ = F_in;
    H_ = H_in;
}
Eigen::VectorXd EKF::getEstimation(){
    return x_;
}
