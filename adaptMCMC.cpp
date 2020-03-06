
//CHECK FOR INT OVERFLOW
//CHANGE LOOP INDEX TO TYPE SIZE_T

#include <iostream>
#include <chrono>
#include <fstream>
#include "BayesModel.h"
#include "Rand.h"
#include "AMHsampler.h"

double fz(double x)
{
	return 0.5 * log((1.0 + x) / (1.0 - x));
}

double fz_inv(double x)
{
	return (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0);
}

double my_exp(double x) // the functor we want to apply
{
	return std::exp(x);
}

double my_log(double x) // the functor we want to apply
{
	return std::log(x);
}

double ldnorm(const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma)
{
	return (-sigma.unaryExpr(&my_log) - 0.5 * ((x - mu).cwiseQuotient(sigma)).cwiseAbs2()).sum();
}


double loglik(Eigen::VectorXd& para, Eigen::MatrixXd& data)
{
	double ll = 0;
	for (size_t i = 0; i < static_cast<size_t>(para.size()); ++i)
	{
		ll += para(i);
	}
	return ll;
}




//normal density with parameters mu and log(sigma)
double mylp(const Eigen::VectorXd& para, const Eigen::MatrixXd& data, const Eigen::VectorXd& para_no_update)
{
	return -data.rows() * para(1) - 0.5 * ((data.col(0) - para(0) * Eigen::MatrixXd::Ones(data.rows(), 1)) * 1 / exp(para(1))).cwiseAbs2().sum();
}


//normal density with parameters mu and log(sigma)
double mylp_mu(const Eigen::VectorXd& para, const Eigen::MatrixXd& data, const Eigen::VectorXd& para_no_update)
{
	return -data.rows() * para_no_update(0) - 0.5 * ((data.col(0) - para(0) * Eigen::MatrixXd::Ones(data.rows(), 1)) * 1 / exp(para_no_update(0))).cwiseAbs2().sum();
}

double mylp_sigma(const Eigen::VectorXd& para, const Eigen::MatrixXd& data, const Eigen::VectorXd& para_no_update)
{
	return -data.rows() * para(0) - 0.5 * ((data.col(0) - para_no_update(0) * Eigen::MatrixXd::Ones(data.rows(), 1)) * 1 / exp(para(0))).cwiseAbs2().sum();
}


int main()
{
	Eigen::MatrixXd data(2000, 1);
	Eigen::VectorXd para(2);
	Rand rand(23);
	para(0) = 0;
	para(1) = 1;
	for (int i = 0; i < data.rows(); ++i)
	{
		data.row(i) = -2 * Eigen::VectorXd::Ones(1) + rand.n();
	}

	BayesModel bm(para, data, mylp);
	
	int iter = 100000;
	AMHSampler mySampler(bm, iter, 2, 13, 0.5);

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000; ++i)
	{
		mySampler.update();
	}

	mySampler.set_adapt_flag(true);

	for (int i = 0; i < iter - 1000; ++i)
	{
		mySampler.update();
	}

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "time taken: " << iter << " iterations took " << duration.count() << " seconds" << std::endl;
	std::cout << "posterior mean estimates: " << mySampler.post_mean(2000) << std::endl;
	//std::cout << "avg accept tail: " << mySampler.get_accept_vec().tail(1000).mean() << std::endl;
	std::cout << "avg accept total: " << mySampler.get_accept_vec().mean() << std::endl;
	


	Eigen::VectorXd para_mu(1);
	Eigen::VectorXd para_sigma(1);
	para_mu(0) = 0;
	para_sigma(0) = 0;
	BayesModel bm_mu(para_mu, data, mylp_mu, para_sigma);
	BayesModel bm_sigma(para_sigma, data, mylp_sigma, para_mu);

	AMHSampler SamplerMu(bm_mu, iter, 2, 123, 0.5);
	AMHSampler SamplerSigma(bm_sigma, iter, 2, 123, 0.5);



	start = std::chrono::high_resolution_clock::now();
	
	
	for (int i = 0; i < 10; ++i)
	{
		SamplerMu.update();
		bm_sigma.set_para_no_update(bm_mu.get_para());
		SamplerSigma.update();
		bm_mu.set_para_no_update(bm_sigma.get_para());
	}

	SamplerMu.set_adapt_flag(true);
	SamplerSigma.set_adapt_flag(true);

	for (int i = 0; i < iter-10; ++i)
	{
		SamplerMu.update();
		bm_sigma.set_para_no_update(bm_mu.get_para());
		SamplerSigma.update();
		bm_mu.set_para_no_update(bm_sigma.get_para());
	}


	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Results for the Gibbs approach:" << std::endl;
	std::cout << "time taken: " << iter << " iterations took " << duration.count() << " seconds" << std::endl;
	std::cout << "posterior mean estimates: " << SamplerMu.post_mean(500) << ", " << SamplerSigma.post_mean(500) << std::endl;
	std::cout << "avg accept total: " << SamplerMu.get_accept_vec().mean() << ", " << SamplerSigma.get_accept_vec().mean() << std::endl;
	return 0;
}

