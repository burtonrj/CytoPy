#include "logicle.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>

struct logicle_params
{
    double T, W, M, A;

    double a, b, c, d, f;
    double w, x0, x1, x2;

    double xTaylor;
    double *taylor;

    double inverse;  // for hyperlog only

    double *lookup;
    int bins;
};

const int TAYLOR_LENGTH = 16;

double solve (double b, double w) {

	double delta;
	double df, t;

	// w == 0 means its really arcsinh
	if (w == 0)
		return b;

	// precision is the same as that of b
	double tolerance = 2 * b * DBL_EPSILON;

	// based on RTSAFE from Numerical Recipes 1st Edition
	// bracket the root
	double d_lo = 0;
	double d_hi = b;

	// bisection first step
	double d = (d_lo + d_hi) / 2;
	double last_delta = d_hi - d_lo;

	// evaluate the f(w,b) = 2 * (ln(d) - ln(b)) + w * (b + d)
	// and its derivative
	double f_b = -2 * log(b) + w * b;
	double f = 2 * log(d) + w * d + f_b;
	double last_f = NAN;

	for (int i = 1; i < 40; ++i)
	{
		// compute the derivative
		df = 2 / d + w;

		// if Newton's method would step outside the bracket
		// or if it isn't converging quickly enough
		if (((d - d_hi) * df - f) * ((d - d_lo) * df - f) >= 0
			|| fabs(1.9 * f) > fabs(last_delta * df))
		{
			// take a bisection step
			delta = (d_hi - d_lo) / 2;
			d = d_lo + delta;
			if (d == d_lo)
				return d; // nothing changed, we're done
		}
		else
		{
			// otherwise take a Newton's method step
			delta = f / df;
			t = d;
			d -= delta;
			if (d == t)
				return d; // nothing changed, we're done
		}
		// if we've reached the desired precision we're done
		if (fabs(delta) < tolerance)
			return d;
		last_delta = delta;

		// recompute the function
		f = 2 * log(d) + w * d + f_b;
		if (f == 0 || f == last_f)
			return d; // found the root or are not going to get any closer
		last_f = f;

		// update the bracketing interval
		if (f < 0)
			d_lo = d;
		else
			d_hi = d;
	}
}

double seriesBiexponential (struct logicle_params p, double scale) {
	// Taylor series is around x1
	double x = scale - p.x1;
	// note that taylor[1] should be identically zero according
	// to the Logicle condition so skip it here
	double sum = p.taylor[TAYLOR_LENGTH - 1] * x;
	for (int i = TAYLOR_LENGTH - 2; i >= 2; --i)
		sum = (sum + p.taylor[i]) * x;
	return (sum * x + p.taylor[0]) * x;
}

double scale (struct logicle_params p, double value) {
	// handle true zero separately
	if (value == 0)
		return p.x1;

	// reflect negative values
	bool negative = value < 0;
	if (negative)
		value = -value;

	// initial guess at solution
	double x;
	if (value < p.f)
		// use linear approximation in the quasi linear region
		x = p.x1 + value / p.taylor[0];
	else
		// otherwise use ordinary logarithm
		x = log(value / p.a) / p.b;

	// try for double precision unless in extended range
	double tolerance = 3 * DBL_EPSILON;
	if (x > 1)
		tolerance = 3 * x * DBL_EPSILON;

	for (int i = 0; i < 40; ++i)
	{
		// compute the function and its first two derivatives
		double ae2bx = p.a * exp(p.b * x);
		double ce2mdx = p.c / exp(p.d * x);
		double y;
		if (x < p.xTaylor)
			// near zero use the Taylor series
			y = seriesBiexponential(p, x) - value;
		else
			// this formulation has better round-off behavior
			y = (ae2bx + p.f) - (ce2mdx + value);
		double abe2bx = p.b * ae2bx;
		double cde2mdx = p.d * ce2mdx;
		double dy = abe2bx + cde2mdx;
		double ddy = p.b * abe2bx - p.d * cde2mdx;

		// this is Halley's method with cubic convergence
		double delta = y / (dy * (1 - y * ddy / (2 * dy * dy)));
		x -= delta;

		// if we've reached the desired precision we're done
		if (fabs(delta) < tolerance) {
			// handle negative arguments
			if (negative)
				return 2 * p.x1 - x;
			else
				return x;
		}
	}

	// TODO: do something if we get here, scale did not converge
};


void logicle_scale(double T, double W, double M, double A, double* x, int n) {
	// allocate the parameter structure
	struct logicle_params p;
	p.taylor = 0;

    // TODO: move these checks to Python
//	if (T <= 0)
//		throw IllegalParameter("T is not positive");
//	if (W <= 0)
//		throw IllegalParameter("W is not positive");
//	if (M <= 0)
//		throw IllegalParameter("M is not positive");
//	if (2 * W > M)
//		throw IllegalParameter("W is too large");
//	if (-A > W || A + W > M - W)
//		throw IllegalParameter("A is too large");

	// standard parameters
	p.T = T;
	p.M = M;
	p.W = W;
	p.A = A;

	// actual parameters
	// formulas from bi-exponential paper
	p.w = W / (M + A);
	p.x2 = A / (M + A);
	p.x1 = p.x2 + p.w;
	p.x0 = p.x2 + 2 * p.w;
	p.b = (M + A) * log(10.);
	p.d = solve(p.b, p.w);
	double c_a = exp(p.x0 * (p.b + p.d));
	double mf_a = exp(p.b * p.x1) - c_a / exp(p.d * p.x1);
	p.a = T / ((exp(p.b) - mf_a) - c_a / exp(p.d));
	p.c = c_a * p.a;
	p.f = -mf_a * p.a;

	// use Taylor series near x1, i.e., data zero to
	// avoid round off problems of formal definition
	p.xTaylor = p.x1 + p.w / 4;

	// compute coefficients of the Taylor series
	double posCoef = p.a * exp(p.b * p.x1);
	double negCoef = -p.c / exp(p.d * p.x1);

	// 16 is enough for full precision of typical scales
	double tmp_taylor[16];
	p.taylor = tmp_taylor;

	for (int i = 0; i < TAYLOR_LENGTH; ++i)
	{
		posCoef *= p.b / (i + 1);
		negCoef *= -p.d / (i + 1);
		(p.taylor)[i] = posCoef + negCoef;
	}
	p.taylor[1] = 0; // exact result of Logicle condition

	// end original initialize method

	for(int j=0;j<n;j++) {
		x[j] = scale(p, x[j]);
	}
}


double hyperscale (struct logicle_params p, double value) {
	// handle true zero separately
	if (value == 0)
		return p.x1;

	// reflect negative values
	bool negative = value < 0;
	if (negative)
		value = -value;

	// initial guess at solution
	double x;
	if (value < p.inverse)
		x = p.x1 + value * p.w / p.inverse;
	else
		// otherwise use ordinary logarithm
		x = log(value / p.a) / p.b;

	// try for double precision unless in extended range
	double tolerance = 3 * DBL_EPSILON;

	for (int i = 0; i < 10; ++i)
	{
		double ae2bx = p.a * exp(p.b * x);
		double y;
		if (x < p.xTaylor)
			// near zero use the Taylor series
			y = seriesBiexponential(p, x) - value;
		else
			// this formulation has better round-off behavior
			y = (ae2bx + p.c * x) - (p.f + value);

		double abe2bx = p.b * ae2bx;
		double dy = abe2bx + p.c;
		double ddy = p.b * abe2bx;

		// this is Halley's method with cubic convergence
		double delta = y / (dy * (1 - y * ddy / (2 * dy * dy)));
		x -= delta;

		// if we've reached the desired precision we're done
		if (fabs(delta) < tolerance) {
			// handle negative arguments
			if (negative)
				return 2 * p.x1 - x;
			else
				return x;
		}
	}

	// TODO: do something if we get here, scale did not converge
};


void hyperlog_scale(double T, double W, double M, double A, double* x, int n) {
	// allocate the parameter structure
	struct logicle_params p;

	// standard parameters
	p.T = T;
	p.M = M;
	p.W = W;
	p.A = A;

	// actual parameters
	p.w = W / (M + A);
	p.x2 = A / (M + A);

	p.x1 = p.x2 + p.w;
	p.x0 = p.x2 + 2 * p.w;

	p.b = (M + A) * log(10);
	double e0 = exp(p.b * p.x0);

	double c_a = e0 / p.w;
	double f_a = exp(p.b * p.x1) + c_a * p.x1;
	p.a = T / (exp(p.b) + c_a - f_a);

	p.c = c_a * p.a;
	p.f = f_a * p.a;

	// use Taylor series near x1, i.e., data zero to
	// avoid round off problems of formal definition
	p.xTaylor = p.x1 + p.w / 4;

	// compute coefficients of the Taylor series
	double coef = p.a * exp(p.b * p.x1);

	// 16 is enough for full precision of typical scales
	double tmp_taylor[16];
	p.taylor = tmp_taylor;

	for (int i = 0; i < TAYLOR_LENGTH; ++i)
	{
		coef *= p.b / (i + 1);
		(p.taylor)[i] = coef;
	}

	p.taylor[0] += p.c;

	bool is_negative = p.x0 < p.x1;
	double tmp_x0;
	if (is_negative) {
	    tmp_x0 = 2 * p.x1 - p.x0;
	} else {
	    tmp_x0 = p.x0;
	}

	if (tmp_x0 < p.xTaylor) {
	    p.inverse = seriesBiexponential(p, tmp_x0);
	} else {
	    p.inverse = (p.a * exp(p.b * tmp_x0) + p.c * tmp_x0);
	}

	if (is_negative) {
	    p.inverse = -p.inverse;
	}

	for(int j = 0; j < n; j++) {
		x[j] = hyperscale(p, x[j]);
	}
}