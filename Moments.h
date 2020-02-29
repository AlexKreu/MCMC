#ifndef MOMENTS_H
#define MOMENTS_H

class Moments
{
private:
	int_fast32_t m_n; //number of data points used to calculate the first empirical covariance matrix
	int_fast32_t m_d;
	double m_n_running; //number of data points including updates, double to perform numerics

public:
	Eigen::VectorXd m_mean;
	Eigen::MatrixXd m_covmat;

	Moments()
	{
		m_n = 0;
		m_n_running = 0;
		m_d = 0;
		m_mean = Eigen::VectorXd::Zero(1);
		m_covmat = Eigen::MatrixXd::Zero(1, 1);
	}

	Moments(const Eigen::MatrixXd& data)
	{
		m_n = data.rows();
		m_n_running = m_n;
		m_d = data.cols();
		Eigen::MatrixXd data_c(m_n, m_d);
		m_mean = data.colwise().mean();
		data_c = data.rowwise() - m_mean.transpose();
		m_covmat = (data_c.transpose() * data_c) / (static_cast<double>(m_n) - 1);
	};

	void update(const Eigen::VectorXd& x)
	{
		m_covmat = (m_n_running - 1.0) / m_n_running * m_covmat +
			1.0 / (m_n_running + 1.0) * (x - m_mean) * (x - m_mean).transpose(); //check them again!!!!
		m_mean = (m_n_running * m_mean + x) / (m_n_running + 1.0);
		++m_n_running;
	}
};

#endif
