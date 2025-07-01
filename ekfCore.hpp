# include <Eigen/Dense>
# include <vector>

class EKF
{
public:
    EKF(const Eigen::VectorXd &x_in, const Eigen::MatrixXd &S_in, const double &dt);
    ~EKF();

    void predict(Eigen::MatrixXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u, const double eps);
    void predict(const Eigen::VectorXd &u, const Eigen::MatrixXd &F_);
    void update(Eigen::MatrixXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u,const Eigen::VectorXd &z, const double eps);
    void update(const Eigen::VectorXd &z, const Eigen::MatrixXd &H_);
    Eigen::MatrixXd numJacobian(Eigen::MatrixXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&), const Eigen::VectorXd &x, const Eigen::VectorXd &u, const double &h);

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


EKF::EKF(const Eigen::VectorXd &x_in, const Eigen::MatrixXd &S_in, const double &dt){
    dt_ = dt;
    x_ = x_in;
    S_ = S_in;
    std::cout<<"EKF initialized"<<std::endl;
}

EKF::~EKF(){
    std::cout<<"EKF destructed"<<std::endl;
}

void EKF::predict(Eigen::MatrixXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u, const double eps){
    // numerical jacobian method
    if(init_flag_){
        Eigen::VectorXd dx_ = func(x_, u);
        x_ = x_ + dx_ * dt_;
        F_ = numJacobian(func, x_, u, eps);
        Eigen::MatrixXd N = S_.transpose() * F_.transpose();
        Eigen::MatrixXd M(N.rows() + Rq_.rows(), N.cols()); 
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
        S_ = qr.matrixQR().triangularView<Eigen::Upper>();
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
        Eigen::MatrixXd N = S_.transpose() * F_.transpose();
        Eigen::MatrixXd M(N.rows() + Rq_.rows(), N.cols()); 
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
        S_ = qr.matrixQR().triangularView<Eigen::Upper>();
    }
    else{
        init_flag_ = true;
    }
    
}

void EKF::update(Eigen::MatrixXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&),const Eigen::VectorXd &u, const Eigen::VectorXd &z, const double eps){
    // numerical jacobian method
    H_ = numJacobian(func, x_, u, eps);
    Eigen::MatrixXd N = S_.transpose() * H_.transpose();
    Eigen::MatrixXd M(N.rows() + Rq_.rows(), N.cols()); 
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
	Eigen::MatrixXd Sy = qr.matrixQR().triangularView<Eigen::Upper>();
    Eigen::MatrixXd P = S_.transpose() * S_;
    Eigen::MatrixXd Pxy = P * H_.transpose();
    Eigen::VectorXd K = Pxy * (Sy.transpose() * Sy).inverse();
    Eigen::VectorXd y = z - H_ * x_;
    x_ = x_ + K * y;
    S_ = (Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K * H_) * P;
}

void EKF::update(const Eigen::VectorXd &z, const Eigen::MatrixXd &H_in){
    // pre-calculated explicit jacobian
    H_ = H_in;
    Eigen::MatrixXd N = S_.transpose() * H_.transpose();
    Eigen::MatrixXd M(N.rows() + Rq_.rows(), N.cols()); 
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
	Eigen::MatrixXd Sy = qr.matrixQR().triangularView<Eigen::Upper>();
    Eigen::MatrixXd P = S_.transpose() * S_;
    Eigen::MatrixXd Pxy = P * H_.transpose();
    Eigen::VectorXd K = Pxy * (Sy.transpose() * Sy).inverse();
    Eigen::VectorXd y = z - H_ * x_;
    x_ = x_ + K * y;
    S_ = (Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K * H_) * P;
}

Eigen::MatrixXd numJacobian(Eigen::MatrixXd (*func)(const Eigen::VectorXd&, const Eigen::VectorXd&), const Eigen::VectorXd &x, const Eigen::VectorXd &u, const double &h){
    Eigen::VectorXd Z = func(x, u);
    Eigen::MatrixXd J(Z.size(), Z.size());
    J<< 0,0,
        0,0;
    for(int i = 0; i < Z.size(); i++){
        Eigen::VectorXd x_plus_h = x;
        double eps = std::max(h, h*std::fabs(x_plus_h(i)));
        x_plus_h(i) += eps;
        Eigen::VectorXd Z_plus = func(x_plus_h, x);
        J.col(i) = (Z_plus - Z) / eps;
    }

    return J;
}
