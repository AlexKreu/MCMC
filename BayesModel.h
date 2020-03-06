#ifndef BAYESMODEL_H
#define BAYESMODEL_H

#include <Eigen/Dense>

class BayesModel
{
	
	typedef double (*LOGPOST)(const Eigen::VectorXd& para, const Eigen::MatrixXd& data, const Eigen::VectorXd& para_no_update);

private:
	Eigen::VectorXd m_para;
	Eigen::MatrixXd m_data;
	LOGPOST m_logpost;
	Eigen::VectorXd m_para_no_update;
	int m_d;

public:
	BayesModel(Eigen::VectorXd& para, Eigen::MatrixXd& data, LOGPOST logpost)
	{
		m_para = para;
		m_data = data;
		m_logpost = logpost;
		m_para_no_update = Eigen::VectorXd::Zero(1);
		m_d = m_para.size();
	}

	BayesModel(Eigen::VectorXd& para, Eigen::MatrixXd& data, LOGPOST logpost, Eigen::VectorXd& para_no_update)
	{
		m_para = para;
		m_data = data;
		m_logpost = logpost;
		m_para_no_update = para_no_update;
		m_d = m_para.size();
	}

	BayesModel()
	{
		m_para = Eigen::VectorXd(1);
		m_data = Eigen::MatrixXd(1, 1);
	}

	double lp()
	{
		return m_logpost(m_para, m_data, m_para_no_update);
	}

	double lp(const Eigen::VectorXd& para)
	{
		return m_logpost(para, m_data, m_para_no_update);
	}

	double lp(const Eigen::VectorXd& para, const Eigen::VectorXd& para_no_update)
	{
		return m_logpost(para, m_data, para_no_update);
	}

	int get_para_dim()
	{
		return m_d;
	}

	void set_para(const Eigen::VectorXd& para)
	{
		m_para = para;
	}

	Eigen::VectorXd get_para()
	{
		return m_para;
	}

	Eigen::VectorXd get_para_no_update()
	{
		return m_para_no_update;
	}

	void set_para_no_update(const Eigen::VectorXd& para_no_update)
	{
		m_para_no_update = para_no_update;
	}

	void set_data(Eigen::MatrixXd data)
	{
		m_data = data;
	}
};

#endif



