#ifndef AMHSAMPLER_H
#define AMHSAMPLER_H

#include <boost/math/distributions/normal.hpp>
#include "BayesModel.h"
#include "Rand.h"
#include "Moments.h"

constexpr double pi = 3.14159265358979323846;

class AMHSampler
{
private:
	BayesModel& m_bm;
	Moments m_moments;
	int m_iter;       //total number of iterations
	int m_iter_count; //the current iteration
	int m_seed;
	int m_d;         //dimension of the parameter vector
	Eigen::MatrixXd m_identity = Eigen::MatrixXd::Identity(m_d, m_d);
	Eigen::MatrixXd m_scaling;
	Rand m_rand;
	Eigen::VectorXd m_para_proposed;
	Eigen::VectorXd m_zero_vec;
	void (AMHSampler::* update_scaling) ();
	Eigen::VectorXd(AMHSampler::* gen_normal) ();
	double m_scaling_para;
	double m_R;
	Eigen::VectorXd m_accept_vec;
	double m_p;
	double m_c_monroe;

public:
	Eigen::MatrixXd m_samples;

	AMHSampler(BayesModel& bm, int iter, double scaling_para = 1, int seed = 123, double p = 0.23) : m_bm(bm)
	{
		m_iter_count = 0;
		m_iter = iter;
		m_para_proposed = m_bm.get_para();
		m_d = bm.get_para_dim();
		m_identity = Eigen::MatrixXd::Identity(m_d, m_d);
		m_scaling_para = scaling_para;
		m_p = p;
		m_c_monroe = calc_c_monroe(m_p, m_d);

		if (m_d == 1)
		{
			m_scaling = Eigen::MatrixXd::Identity(m_d, m_d) * m_scaling_para;
		}
		else {
			m_scaling = Eigen::MatrixXd::Identity(m_d, m_d) * pow(m_scaling_para, 2);
		}

		m_rand.set_seed(seed);
		m_samples = Eigen::MatrixXd(m_iter, m_d);
		set_adapt_flag(false);
		update_scaling = &AMHSampler::no_update;
		m_R = -1000;
		m_accept_vec = Eigen::VectorXd(m_iter);
		m_zero_vec = Eigen::VectorXd::Zero(m_d);

		if (m_d == 1)
		{
			gen_normal = &AMHSampler::gen_normal_univariate;
		}
		else {
			gen_normal = &AMHSampler::gen_normal_multivariate;
		}
	}

	static double calc_c_monroe(const double p, const double d)
	{
		double alpha;
		boost::math::normal norm;
		alpha = -boost::math::quantile(norm, p / 2.0);
		return (1.0 - 1.0 / d) * (sqrt(2 * pi) * exp(alpha / 2.0)) / (2 * alpha) + 1.0 / (1.0 * p * (1.0 - p));
	}

	void update_scaling_univariate()
	{
		m_scaling_para = exp(log(m_scaling_para) + m_c_monroe * (m_R - m_p) / double(m_iter_count));
		m_scaling(0, 0) = m_scaling_para; //this is the standard deviation
	}

	void update_scaling_multivariate()
	{
		m_scaling_para = exp(log(m_scaling_para) + m_c_monroe * (m_R - m_p) / (std::max(200, (m_iter_count + 1) / m_d)));
		m_moments.update(m_bm.get_para());
		m_scaling = m_scaling_para * m_scaling_para * (m_moments.m_covmat +
		m_scaling_para * m_scaling_para / double(m_iter_count + 1.0) * m_identity);
	}

	void no_update()
	{
	}

	void set_adapt_flag(bool adapt)
	{
		if (adapt)
		{
			if(m_iter_count < 1) throw "m_iter_count > 0 is required to turn on adaption";
			m_moments = Moments(m_samples.block(0, 0, m_iter_count, m_d));
			if (m_d == 1)
			{
				update_scaling = &AMHSampler::update_scaling_univariate;
			}
			else {
				update_scaling = &AMHSampler::update_scaling_multivariate;
			}
		}
		else {
			update_scaling = &AMHSampler::no_update;
		}
	}

	Eigen::VectorXd gen_normal_univariate()
	{
		return m_rand.n(m_zero_vec, m_scaling);
	}

	Eigen::VectorXd gen_normal_multivariate()
	{
		return m_rand.mn(m_zero_vec, m_scaling);
	}

	void update()
	{
		m_para_proposed = m_bm.get_para() + (this->*gen_normal)();
		m_R = std::min(exp(m_bm.lp(m_para_proposed) - m_bm.lp()), 1.0);
		if (m_R > m_rand.unif())
		{
			m_bm.set_para(m_para_proposed);
		}
		m_samples.row(m_iter_count) = m_bm.get_para();
		(this->*update_scaling)();
		m_accept_vec(m_iter_count) = m_R;
		++m_iter_count;
	}

	Eigen::VectorXd get_accept_vec()
	{
		return m_accept_vec;
	}

	Eigen::MatrixXd get_samples()
	{
		return m_samples;
	}

	Eigen::VectorXd post_mean(int burnin = 0)
	{
		assert(burnin < m_iter_count);
		return m_samples.block(burnin, 0, m_iter_count - burnin, m_d).colwise().mean();
	}
};

#endif

