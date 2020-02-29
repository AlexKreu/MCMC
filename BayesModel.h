#ifndef BAYESMODEL_H
#define BAYESMODEL_H

#include <Eigen/Dense>

class BayesModel
{
private:
	Eigen::VectorXd m_para;
	Eigen::MatrixXd m_data;

public:
	double (*m_logpost)(Eigen::VectorXd& para, Eigen::MatrixXd& data);

	BayesModel(Eigen::VectorXd& para, Eigen::MatrixXd& data, double (*logpost)(Eigen::VectorXd& para, Eigen::MatrixXd& data))
	{
		m_para = para;
		m_data = data;
		m_logpost = logpost;
	}

	BayesModel()
	{
		m_para = Eigen::VectorXd(1);
		m_data = Eigen::MatrixXd(1, 1);
	}

	double lp()
	{
		return (*m_logpost)(m_para, m_data);
	}

	double lp(Eigen::VectorXd para)
	{
		return (*m_logpost)(para, m_data);
	}

	int get_para_dim()
	{
		return m_para.size();
	}

	Eigen::VectorXd get_para()
	{
		return m_para;
	}

	void set_data(Eigen::MatrixXd data)
	{
		m_data = data;
	}
};

#endif



