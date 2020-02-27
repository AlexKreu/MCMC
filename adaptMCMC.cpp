// Project1.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//
//CHECK FOR INT OVERFLOW
//CHANGE LOOP INDEX TO TYPE SIZE_T

#include <iostream>
#include <vector>
#include <cmath>
//#include <math.h>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <chrono>
#include <Eigen/Dense>
#include <boost/math/distributions/normal.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <fstream>
#include <Eigen/LU> 



constexpr double pi = 3.14159265358979323846;
typedef int_fast32_t fint;

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

double lp_svmodel(Eigen::VectorXd& para, Eigen::MatrixXd& data)
{
	int n = data.cols();
	//return ldnorm_sv(data.col(0), (para.segment(4, n) * 0.5));

	return ldnorm(data.col(0).segment(0, n), Eigen::VectorXd::Zero(n), (0.5 * para.segment(4, n)).unaryExpr(&my_exp))
		+ ldnorm(para.segment(4, n), para(0) * Eigen::VectorXd::Ones(n)
			+ fz_inv(para(1)) * (para.segment(3, n) - para(0) * Eigen::VectorXd::Ones(n)), exp(para(2)) * Eigen::VectorXd::Ones(n));
}

class Point2d
{
private:
	double m_x;
	double m_y;
	int m_test;

public:
	Point2d(double x = 0.0, double y = 0.0)
	{
		m_x = x;
		m_y = y;

	}

	void print()
	{
		std::cout << "Point2d(" << m_x << "," << m_y << ")\n";
	}

	double distanceTo(Point2d p)
	{
		return(sqrt((m_x - p.m_x) * (m_x - p.m_x) + (m_y - p.m_y) * (m_y - p.m_y)));
	}

	friend double distanceFrom(Point2d p1, Point2d p2);

	int get_test()
	{
		return m_test;
	}
};

double distanceFrom(Point2d p1, Point2d p2)
{
	return(sqrt((p1.m_x - p2.m_x) * (p1.m_x - p2.m_x) + (p1.m_y - p2.m_y) * (p1.m_y - p2.m_y)));
}

myMat test(int ncols, int nrows)
{
	myMat mat(ncols, nrows);
	return(mat);
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

double calc_c_monroe(double p, double d)
{
	double alpha;
	boost::math::normal norm;
	alpha = -boost::math::quantile(norm, p / 2.0);
	return (1.0 - 1.0 / d) * (sqrt(2 * pi) * exp(alpha / 2.0)) / (2 * alpha) + 1.0 / (1.0 * p * (1.0 - p));
}

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

	double ll()
	{
		return (*m_logpost)(m_para, m_data);
	}

	double ll(Eigen::VectorXd para)
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

	//erster block geht bis erste zhl minus 1, hat dann größe gleich der ersten zahl, starte mit 0

	Eigen::MatrixXd sample_RWMH(int iter, int seed, std::vector<int> blocks_vector)
	{
		int d = m_para.size();
		int n_blocks = blocks_vector.size() - 1;


		std::vector<int> block_size(d);
		std::vector<std::vector<int>> blocks(d);
		std::vector<Eigen::MatrixXd> scaling(d);
		Eigen::VectorXd scaling_uni(n_blocks);
		scaling_uni = Eigen::VectorXd::Ones(n_blocks);
		std::vector<Eigen::VectorXd> runmean(d);
		std::vector<Eigen::MatrixXd> runvar(d);
		std::vector<Eigen::MatrixXd> tmp_mats(d);
		std::vector<Eigen::VectorXd> normals(d);
		std::vector<int> tmp(d);

		double R;
		for (int i = 0; i < n_blocks; ++i)
		{
			block_size.at(i) = blocks_vector.at(i + 1) - blocks_vector.at(i);
			scaling.at(i) = Eigen::MatrixXd::Identity(block_size.at(i), block_size.at(i));
			tmp_mats.at(i) = Eigen::MatrixXd::Identity(block_size.at(i), block_size.at(i));
			runvar.at(i) = Eigen::MatrixXd::Identity(block_size.at(i), block_size.at(i));
			runmean.at(i) = Eigen::VectorXd(block_size.at(i));
			normals.at(i) = Eigen::VectorXd(block_size.at(i));
		}


		Eigen::MatrixXd samples(iter, d);
		Eigen::VectorXd para_proposed(d);
		Eigen::VectorXd para_current(d);
		boost::mt19937 randgen(seed);
		boost::uniform_real<> uniform_dist(0, 1);
		boost::normal_distribution<double> standard_normal_dist(0, 1);
		boost::variate_generator<boost::mt19937&, boost::uniform_real<> > sample_uniform(randgen, uniform_dist);
		boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > sample_standard_normal(randgen, standard_normal_dist);
		para_current = m_para;
		samples.row(0) = para_current;

		int warmup = 100;
		std::vector<int> start_iter(2);
		std::vector<int> end_iter(2);

		start_iter.at(0) = 1;
		start_iter.at(1) = warmup;
		end_iter.at(0) = warmup;
		end_iter.at(1) = iter;

		/*
		for(int warmup_ind=0; warmup_ind<2; warmup_ind++)
		{
			for (int it = start_iter.at(warmup_ind); it < end_iter.at(warmup_ind); it++)
			{
				for (int i_block = 0; i_block < n_blocks; i_block++)
				{
					Eigen::LLT<Eigen::MatrixXd> lltOfS(scaling.at(i_block));

					for (int j = 0; j < block_size.at(i_block); j++)
					{
						normals.at(i_block)(j) = sample_standard_normal();
					}

					para_proposed = para_current;
					para_proposed.segment(blocks_vector.at(i_block), block_size.at(i_block)) = para_current.segment(blocks_vector.at(i_block), block_size.at(i_block)) + lltOfS.matrixL() * normals.at(i_block);

					R = ll(para_proposed) - ll(para_current);
					if (R > log(sample_uniform()))
					{
						para_current = para_proposed;
					}
					if (warmup_ind == 1)
					{
						runmean.at(i_block) = (it * runmean.at(i_block) + para_current.segment(blocks_vector.at(i_block), block_size.at(i_block))) / (it + 1);

						if (block_size.at(i_block) == 1)
						{
							scaling.at(i_block)(0,0) = exp(log(scaling.at(i_block)(0, 0)) + 4.058442 * (std::min(exp(R), 1.0) - 0.44) / (it + 1));
						}
						else
						{
							runvar.at(i_block) = (it - 1) / it * runvar.at(i_block) + 1 / (it + 1) * (para_current.segment(blocks_vector.at(i_block), block_size.at(i_block)) - runmean.at(i_block)) * (para_current.segment(blocks_vector.at(i_block), block_size.at(i_block)) - runmean.at(i_block)).transpose();
							scaling_uni(i_block) = exp(log(scaling_uni(i_block)) + 6.59795 * (std::min(exp(R), 1.0) - 0.23) / (std::max(200, (it + 1) / 2)));
							scaling.at(i_block) = scaling_uni(i_block) * scaling_uni(i_block) * (runvar.at(i_block) + scaling_uni(i_block) * scaling_uni(i_block) / (it + 1) * Eigen::MatrixXd::Identity(block_size.at(i_block), block_size.at(i_block)));
						}
					}
				}
				samples.row(it) = para_current;
			}

			if (warmup_ind == 0)
			{
				for (int i_block = 0; i_block < n_blocks; i_block++)
				{
					if (block_size.at(i_block) > 1)
					{
					//tmp_mats.at(i_block) = samples.block(0, blocks_vector.at(i_block), warmup, block_size.at(i_block));
					//tmp_mats.at(i_block) = tmp_mats.at(i_block).rowwise() - tmp_mats.at(i_block).colwise().mean();
					//runmean.at(i_block) = tmp_mats.at(i_block).colwise().mean();
				//	runvar.at(i_block) = (tmp_mats.at(i_block).transpose() * tmp_mats.at(i_block)) / (warmup - 1);
					//scaling.at(i_block) = runvar.at(i_block);
					}
				}

			}


		}
		*/

		return samples;
	}

};


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

	double sn()
	{
		return m_sngen();
	}

	Eigen::VectorXd n(Eigen::MatrixXd sigma)
	{
		return sigma * m_sngen();
	}

	double unif()
	{
		return m_ugen();
	}

	Eigen::VectorXd mn(Eigen::MatrixXd covmat)
	{
		Eigen::VectorXd normals(covmat.cols());
		Eigen::LLT<Eigen::MatrixXd> llt(covmat);
		for (int i = 0; i < covmat.cols(); ++i)
		{
			normals(i) = sn();
		}
		return llt.matrixL() * normals;
	}

	void set_seed(int seed)
	{
		m_randgen = RENG(seed);
	}
};



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





class MCMCSampler
{
public:
	BayesModel m_bm;
	Moments m_moments;
	int m_iter;
	int m_seed;
	int m_d;
	int m_iter_count;
	Eigen::MatrixXd m_identity;
	Eigen::MatrixXd m_scaling;
	Rand m_rand;
	Eigen::VectorXd m_para_proposed;
	Eigen::VectorXd m_para_current;
	void (MCMCSampler::* update_scaling) ();
	Eigen::VectorXd(MCMCSampler::* gen_normal) ();
	bool m_adapt;
	double m_scaling_para;
	double m_R;
	Eigen::VectorXd m_accept_vec;
	double m_p;
	double m_c_monroe;

public:
	Eigen::MatrixXd m_samples;

	MCMCSampler(BayesModel& bm, int iter, double scaling_para = 1, int seed = 123, double p = 0.23)
	{
		m_iter_count = 0;
		m_iter = iter;
		m_bm = bm;
		m_para_current = m_bm.get_para();
		m_para_proposed = m_para_current;
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
		m_adapt = false;
		set_adapt_flag(m_adapt);
		update_scaling = &MCMCSampler::no_update;
		m_R = -1000;
		m_accept_vec = Eigen::VectorXd(m_iter);

		if (m_d == 1)
		{
			gen_normal = &MCMCSampler::gen_normal_univariate;
		}
		else {
			gen_normal = &MCMCSampler::gen_normal_multivariate;
		}
	}



	void update_scaling_univariate()
	{
		m_scaling_para = exp(log(m_scaling_para) + m_c_monroe * (m_R - m_p) / double(m_iter_count));
		m_scaling(0, 0) = m_scaling_para; //this is the standard deviation
	}

	void update_scaling_multivariate()
	{
		//4.058442
		m_scaling_para = exp(log(m_scaling_para) + m_c_monroe * (m_R - m_p) / (std::max(200, (m_iter_count + 1) / m_d)));
		m_moments.update(m_para_current);
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

			m_moments = Moments(m_samples.block(0, 0, m_iter_count, m_d));

			if (m_d == 1)
			{
				update_scaling = &MCMCSampler::update_scaling_univariate;
			}
			else {
				update_scaling = &MCMCSampler::update_scaling_multivariate;
			}
		}
		else {
			update_scaling = &MCMCSampler::no_update;
		}
	}

	Eigen::VectorXd gen_normal_univariate()
	{
		return m_rand.n(m_scaling);
	}

	Eigen::VectorXd gen_normal_multivariate()
	{
		return m_rand.mn(m_scaling);
	}

	void update()
	{
		m_para_proposed = m_para_current + (this->*gen_normal)();
		m_R = std::min(exp(m_bm.ll(m_para_proposed) - m_bm.ll(m_para_current)), 1.0);
		if (m_R > m_rand.unif())
		{
			m_para_current = m_para_proposed;
		}
		m_samples.row(m_iter_count) = m_para_current;
		(this->*update_scaling)();
		m_accept_vec(m_iter_count) = m_R;
		++m_iter_count;
	}

	Eigen::VectorXd get_accept_vec()
	{
		return m_accept_vec;
	}

};



//normal density with parameters mu, and log(sigma)
double mylp(Eigen::VectorXd& para, Eigen::MatrixXd& data)
{
	return -data.rows() * para(1) - 0.5 * ((data.col(0) - para(0) * Eigen::MatrixXd::Ones(data.rows(), 1)) * 1 / exp(para(1))).cwiseAbs2().sum();;
}

Eigen::VectorXd sim_sv(double mu, double phi, double sigma, int n, int seed = 123)
{
	Rand ra(seed);
	Eigen::VectorXd h(n + 1);
	Eigen::VectorXd data(n);
	h(0) = mu + sigma / sqrt(1 - pow(phi, 2)) * ra.sn();
	for (int i = 1; i < (n + 1); i++)
	{
		h(i) = mu + phi * (h(i - 1) - mu) + sigma * ra.sn();
		data(i - 1) = exp(h(i) / 2.0) * ra.sn();
	}
	return data;
}


int main()
{
	int seed = 2134;
	boost::mt19937 randgen(seed);
	boost::uniform_real<> uniform_dist(0, 1);
	boost::normal_distribution<double> standard_normal_dist(0, 1);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<> > sample_uniform(randgen, uniform_dist);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > sample_standard_normal(randgen, standard_normal_dist);

	//Point2d xx(1, 2);
	//std::cout << "test is:" << xx.get_test();


	int_fast32_t test_int;

	std::cout << sizeof(int_fast32_t) << std::endl;
	test_int = 10000000000;
	std::cout << test_int << std::endl;



	Eigen::MatrixXd data(500, 1);
	Eigen::VectorXd para(2);
	Rand rand(23);
	para(0) = 0;
	para(1) = 1;

	for (int i = 0; i < data.rows(); ++i)
	{
		data(i, 0) = 0 + 1 * rand.sn();
	}



	int xx{ 2 }; // local variable, no linkage

	{
		int xx;
		xx = 3; // this identifier x refers to a different object than the previous x
	}
	std::cout << "xx is: " << xx;
	BayesModel bm(para, data, mylp);
	int iter = 1000;
	MCMCSampler mySampler(bm, iter, 2, 13, 0.23);


	int n_sv = 20;
	Eigen::MatrixXd data_sv(n_sv, 1);
	data_sv.col(0) = sim_sv(0.3, 0, 0.1, n_sv, 1723);
	Eigen::VectorXd para_sv(n_sv + 4);
	para_sv = Eigen::VectorXd::Zero(n_sv + 4);
	int iter_sv = 100000;
	BayesModel bm_sv(para_sv, data_sv, lp_svmodel);
	MCMCSampler mySampler_sv(bm_sv, iter_sv, 0.005);



	//mySampler(MCMCSampler(bm, 44, 123));
	Eigen::VectorXd ONE = Eigen::VectorXd::Ones(4);
	//std::cout << "time taken: " << ldnorm(3.0*ONE, 2.0*ONE, 3.0*ONE) << std::endl;
	//std::cout << "time taken: " << data_sv << std::endl;

	std::cout << "lp_svmodel: " << lp_svmodel(para_sv, data_sv) << std::endl;



	iter = iter_sv;

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000; ++i)
	{
		mySampler_sv.update();
	}

	std::cout << "rows: " << mySampler_sv.m_samples.rows() << std::endl;
	std::cout << "cols: " << mySampler_sv.m_samples.cols() << std::endl;

	mySampler_sv.set_adapt_flag(true);



	for (int i = 0; i < iter - 1000; ++i)
	{
		mySampler_sv.update();
	}


	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	Eigen::VectorXd postmean(2);
	postmean(0) = mySampler_sv.m_samples.col(0).segment(500, iter - 500).mean();
	postmean(1) = mySampler_sv.m_samples.col(1).segment(500, iter - 500).unaryExpr(&my_exp).mean();
	std::cout << "time taken: " << duration.count() << std::endl;
	std::cout << "posterior mean estimates: " << postmean << std::endl;
	std::cout << "avg accept tail: " << mySampler_sv.get_accept_vec().tail(1000).mean() << std::endl;
	std::cout << "avg accept total: " << mySampler_sv.get_accept_vec().mean() << std::endl;




	/*
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; ++i)
	{
		mySampler.update();
	}
	mySampler.set_adapt_flag(true);
	for (int i = 0; i < iter - 10; ++i)
	{
		mySampler.update();
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	Eigen::VectorXd postmean(2);
	postmean(0) = mySampler.m_samples.col(0).segment(500, iter - 500).mean();
	postmean(1) = mySampler.m_samples.col(1).segment(500, iter - 500).unaryExpr(&my_exp).mean();

	std::cout << "time taken: " << duration.count() << std::endl;
	std::cout << "posterior mean estimates: " << postmean << std::endl;
	std::cout << "avg accept tail: " << mySampler.get_accept_vec().tail(1000).mean() << std::endl;
	std::cout << "avg accept total: " << mySampler.get_accept_vec().mean() << std::endl;
	std::cout << "c monroe: " << calc_c_monroe(0.23, 2) << std::endl;
	*/

	std::ofstream myfile;
	myfile.open("example.txt");
	myfile << mySampler.m_samples;
	myfile.close();


	return 0;



	//xx = Point2d(1, 4);
	Rand ra(12223);
	Eigen::MatrixXd co(2, 2);
	Eigen::MatrixXd no_samples(10000, 2);
	co(0, 0) = 1;
	co(1, 1) = 1;
	co(1, 0) = 0;
	co(0, 1) = 0;

	Eigen::LLT<Eigen::MatrixXd> llt(co);
	Eigen::MatrixXd L = llt.matrixL();
	std::cout << L * L.transpose() << std::endl;

	for (int i = 0; i < 10000; ++i)
	{
		no_samples.row(i) = ra.mn(co);
	}

	Eigen::MatrixXd test(3, 2);
	test(0, 0) = 1;
	test(0, 1) = 2;
	test(1, 0) = 2;
	test(1, 1) = 2;
	test(2, 0) = 1;
	test(2, 1) = 4;

	Moments mom10(test);
	std::cout << "test var: " << mom10.m_covmat(0, 0) << " " << mom10.m_covmat(0, 1) << " " << mom10.m_covmat(1, 0) << " " << mom10.m_covmat(1, 1) << std::endl;

	Eigen::MatrixXd test2(4, 2);
	test2.block(0, 0, 3, 2) = test;

	Eigen::VectorXd newv(2);
	newv(0) = 5;
	newv(1) = 6;
	test2.row(3) = newv;
	mom10.update(newv);

	Moments mom11(test2);
	std::cout << "test2: " << test2 << std::endl;

	std::cout << "test var update: " << mom10.m_covmat(0, 0) << " " << mom10.m_covmat(0, 1) << " " << mom10.m_covmat(1, 0) << " " << mom10.m_covmat(1, 1) << std::endl;

	std::cout << "test2 var: " << mom11.m_covmat(0, 0) << " " << mom11.m_covmat(0, 1) << " " << mom11.m_covmat(1, 0) << " " << mom11.m_covmat(1, 1) << std::endl;


	std::cout << "var0: " << 1.0 / 9999.0 * (no_samples.col(0) - no_samples.col(0).mean() * Eigen::VectorXd::Ones(10000)).cwiseAbs2().sum() << std::endl;
	std::cout << "var1: " << 1.0 / 9999.0 * (no_samples.col(1) - no_samples.col(1).mean() * Eigen::VectorXd::Ones(10000)).cwiseAbs2().sum() << std::endl;


	std::cout << "normal samples: " << no_samples.block(0, 0, 15, 2) << std::endl;
	Moments mom(no_samples.block(0, 0, 100, 2));
	std::cout << "mean: " << mom.m_mean << std::endl;
	std::cout << "covmat: " << mom.m_covmat << std::endl;
	Moments mom5(no_samples.block(0, 0, 99, 2));
	mom5.update(no_samples.block(99, 0, 1, 2));
	std::cout << "covmat2: " << mom5.m_covmat << std::endl;


	//Rand ra(11);
	//ra = Rand(12);


	//int ab = 1;
	//int* a = &ab;
	//int b = 3;
	//int& c = b;
	//c = ab;

	//std::cout << b;

	//a = &b;


	//double (*mylp_pointer)(Eigen::VectorXd& para, Eigen::MatrixXd& data);
	//mylp_pointer = mylp;

	para(0) = 0;
	para(1) = 1;

	for (int i = 0; i < data.rows(); ++i)
	{
		data(i, 0) = 0 + sample_standard_normal();
	}









	for (int i = 0; i < 10000; i++)
	{
		no_samples.row(i) = mySampler.gen_normal_multivariate();
	}


	no_samples.col(0) = no_samples.col(0) + Eigen::VectorXd::Ones(10000);
	no_samples.col(1) = no_samples.col(1) + 2 * Eigen::VectorXd::Ones(10000);
	Eigen::MatrixXd data_c(no_samples.rows(), 2);
	Eigen::VectorXd meanv = no_samples.colwise().mean();
	data_c = no_samples.rowwise() - meanv.transpose();
	std::cout << "is zero " << data_c.colwise().mean() << std::endl;


	std::cout << "var0_my: " << 1.0 / 9999.0 * (no_samples.col(0) - no_samples.col(0).mean() * Eigen::VectorXd::Ones(10000)).cwiseAbs2().sum() << std::endl;
	std::cout << "var1_my: " << 1.0 / 9999.0 * (no_samples.col(1) - no_samples.col(1).mean() * Eigen::VectorXd::Ones(10000)).cwiseAbs2().sum() << std::endl;	std::cout << "scalinpmat " << mySampler.m_scaling << std::endl;
	std::cout << "scalinpara " << mySampler.m_scaling_para << std::endl;

	std::cout << "scalinpmat " << mySampler.m_scaling << std::endl;

	std::cout << "normal samples: " << no_samples.block(0, 0, 15, 2) << std::endl;
	Moments mom2(no_samples);
	std::cout << "mean: " << mom2.m_mean << std::endl;
	std::cout << "covmat: " << mom2.m_covmat << std::endl;
	// To get the value of duration use the count() 
	// member function on the duration object 


	return 0;

}

