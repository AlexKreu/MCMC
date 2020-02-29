#ifndef RAND_H
#define RAND_H

#include <Eigen/Dense>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

static const Eigen::VectorXd zero_vec = Eigen::VectorXd::Zero(1);
static const Eigen::MatrixXd one_mat = Eigen::MatrixXd::Ones(1, 1);

class Rand
{
	typedef boost::mt19937                     RENG;    // Mersenne Twister
	typedef boost::normal_distribution<double> NDIST;   // Normal Distribution
	typedef boost::variate_generator<RENG&, NDIST> NGEN;   // Normal Distribution Generator
	typedef boost::uniform_real<double> UNIF;   // Uniform Distribution
	typedef boost::variate_generator<RENG&, UNIF> UGEN; //Uniform Distribution Generator

private:
	RENG m_randgen;
	NDIST m_sndist;
	NGEN m_sngen;
	UNIF m_unif;
	UGEN m_ugen;

public:
	Rand(int seed = 123) : m_randgen(RENG(seed)), m_sndist(NDIST(0, 1)), m_sngen(m_randgen, m_sndist), m_unif(UNIF(0, 1)), m_ugen(m_randgen, m_unif)
	{};

	void set_seed(int seed)
	{
		m_randgen = RENG(seed);
	}

	double unif()
	{
		return m_ugen();
	}

	Eigen::VectorXd n(Eigen::MatrixXd mu = zero_vec, Eigen::MatrixXd sigma = one_mat)
	{
		return mu + sigma * m_sngen();
	}

	Eigen::VectorXd mn(Eigen::VectorXd meanvec, Eigen::MatrixXd covmat)
	{
		Eigen::VectorXd normals(covmat.cols());
		Eigen::LLT<Eigen::MatrixXd> llt(covmat);
		for (int i = 0; i < covmat.cols(); ++i)
		{
			normals(i) = m_sngen();
		}
		return meanvec + llt.matrixL() * normals;
	}
};

#endif


