#include "RNN.h"
#include <iostream>
#include <vector>	
#include <typeinfo>
#include <boost/array.hpp>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace boost;


// debugging tools lol
void print(auto s){ cout << s << endl; }

void print2(auto s1, auto s2){ cout << s1 << " " << s2 << endl; }


/*
RNN OVERVIEW:
Structured as:

[OUTPUT LAYER]
   |
  [V]
   |
[HIDDEN LAYER] --> [W_0] --> [W_1] --> ... --> [W_s]
   |
  [U]
   |
[INPUT LAYER]

with weights as [U, W, V], 
except for LSTM we just make U and W into one weight by stacking [h, X]

---------------------------------------------------------------------------------------------

LSTM OVERVIEW:
-we follow the model as in "http://colah.github.io/posts/2015-08-Understanding-LSTMs/"
-using S() = sigmoid(), T() = tanh(), * = mat mult, ** = element mult, b_k = bias for k gate

f_t = S(W_f*[h_{t-1}, x_t] + b_f)
o_t = S(W_o*[h_{t-1}, x_t] + b_o)
i_t = S(W_i*[h_{t-1}, x_t] + b_i)
cc_t = tanh(W_c*[h_{t-1}, x_t] + b_c)
c_t = f_t**c_{t-1} + i_t**cc_t 
h = o_t**tanh(c_t)

*/


// TODO: Fix sizes of MatrixXd instead of X-dimensional for most places

// IMPORTANT TODO: Need to add cache for all forw_prop updates

RNN::RNN(MatrixXd X, int bpttLen, featureDim, outputDim, hiddenDim): 
bpttLen(bpttLen), 
featureDim(featureDim), 
outputDim(outputDim), 
hiddenDim(hiddenDim), 
Z(hiddenDim + featureDim),
M(X.rows()), 
N(X.cols()), 
X(X)
{
	maxX = X.colwise().maxCoeff();
	minX = X.colwise().minCoeff();
	minMax = maxX - minX;

	// weights
	thetaWInput = MatrixXd::Random(Z, hiddenDim) * (1/sqrt(Z));
	thetaWForget = MatrixXd::Random(Z, hiddenDim) * (1/sqrt(Z));
	thetaWOutput = MatrixXd::Random(Z, hiddenDim) * (1/sqrt(Z));
	thetaWCgate = MatrixXd::Random(Z, hiddenDim) * (1/sqrt(Z));

	biasInput = MatrixXd::Random(1, hiddenDim);
	biasForget = MatrixXd::Random(1, hiddenDim);
	biasOut = MatrixXd::Random(1, hiddenDim);
	biasCgate = MatrixXd::Random(1, hiddenDim);

	thetaV = MatrixXd::Random(hiddenDim, outputDim) * (1/sqrt(hiddenDim));
	biasV = MatrixXd::Random(1, hiddenDim);

}

// TODO: divide prices by one value
void 
RNN::normalize(MatrixXd &m){
	for (int i = 0; i < m.cols(); ++i)
		m.col(i) = (m.col(i).array() - minX(i)) / minMax(i);
}

void 
RNN::denormalize(MatrixXd &m){
	for (int i = 0; i < m.cols(); ++i)
		m.col(i) = (m.col(i).array() * minMax(i)) + minX(i);
}

// for consistency
MatrixXd 
RNN::activationIdentity(MatrixXd m){
	return m;
}

// reLU! f(x) ~= ln(1+e^x)
MatrixXd 
RNN::activationRelU(MatrixXd m){
	return (m.array().exp() + 1).log();
}

// tanh! f(x) = tanh(x)
MatrixXd 
RNN::activationTanh(MatrixXd m){
	return m.array().tanh();
}

// logistic sigmoid!
// f(t) = 1/ 1 + e^(-t)
MatrixXd 
RNN::activationSigmoid(MatrixXd m){
	return (((-1)*m).array().exp() + 1).array().inverse();	
}

MatrixXd 
RNN::gradTanh(MatrixXd m){
	return 1 - m.array().square();
}

MatrixXd 
RNN::gradSigmoid(MatrixXd m){
	return activationSigmoid(m) * 
			((-1.0 * activationSigmoid(m).array()) + 1).matrix();
}

// ??? Eigen sucks ???
MatrixXd 
RNN::gradRelU(MatrixXd m){
	return m;
	// return m.array() > 0.0;
}


// returns (predicted_outs, old_states, caches for interm. values)
// == (y, h, c, cache), cache = [f, i, o, c]
// cache contains cellInput, forget, etc. to be used for BPTT
tuple<MatrixXd, MatrixXd, MatrixXd, array<MatrixXd, 4>> 
RNN::forwardProp(MatrixXd &x, MatrixXd &h_old, MatrixXd &c_old){

	int tSteps = x.rows(); // # examples, should be = 1 for SGD

	// state matrix
	MatrixXd y = MatrixXd::Zero(tSteps, outputDim);

	// 1xZ
	MatrixXd cellForget;
	MatrixXd cellInput;
	MatrixXd cellOutput;
	MatrixXd cellCgate; 

	MatrixXd c;
	MatrixXd h;
	MatrixXd y;

	
	MatrixXd xPrime = MatrixXd::Zero(tSteps, Z); // 1xZ
	xPrime << h_old, x; // x' = [h_{t-1}, x_t] stacked --> 1x(h+n)

	// LSTM stuff
	cellForget = activationSigmoid(xPrime * thetaWForget + biasForget); // 1xZ
	cellInput = activationSigmoid(xPrime * thetaWInput + biasInput);
	cellOutput = activationSigmoid(xPrime * thetaWOutput + biasOut);
	cellCgate = activationTanh(xPrime * thetaWCgate + biasCgate);

	// 1xZ
	c = (cellForget.array() * c_old.array()) + (cellInput.array() * cellCgate.array())

	// 1xZ
	h = cellOutput.array() * activationTanh(c).array();

	// 1xZ
	y = activationIdentity(h * thetaV + biasV); // need linearity for regression

	array<MatrixXd, 2> states{{c, h}};
	array<MatrixXd, 4> cache{{cellForget, cellInput, cellOutput, cellCgate}};

	return make_tuple(y, h, c, cache);
}

MatrixXd 
RNN::predict(MatrixXd &x_test){
	// since we are only predicting stock price, features out is Mx1
	return forwardProp(x_test)[1];
}


double 
RNN::cost(MatrixXd &x, MatrixXd &y, double reg_const){
	double loss = 0;
	int n = x.rows();

	for (int i = 0; i < n; ++i)
	{
		boost::array<MatrixXd, 2> a = forwardProp(x.row(i));
		MatrixXd out = a[1];
		loss += pow(y(i) - out(0), 2);
		// out_i = 1x1, y_i aka y(i, 0) aka y(i) = 1x1
	}

	loss = sqrt(loss/n);

	// TODO:: REGULARIZATION

	return loss;
}


// returns (map_grad, dh_next, dc_next)
tuple< map<string, MatrixXd>, MatrixXd, MatrixXd > 
RNN::backPropTT(MatrixXd &x, MatrixXd &y, 
				MatrixXd dc_next, MatrixXd dh_next){

	int T = y.rows();
	// predicted_outs, old_
	// TODO ADD TO CACHE FROM FPROP
	// xPrime -- c_old (c_{t-1}, c_{t-2}, etc.) -- 
 	tuple<MatrixXd, MatrixXd, MatrixXd, array<MatrixXd, 4>> a = forwardProp(x);
 	MatrixXd outs = get<0>(a);
 	MatrixXd c = get<1>(a);
 	MatrixXd h = get<2>(a);
 	auto cache = get<3>(a);
 	MatrixXd hf = cache[0];
 	MatrixXd hi = cache[1];
 	MatrixXd ho = cache[2];
 	MatrixXd hc = cache[3]; 


	// graidents for weights, lots of them so we use a dict
	map<string, MatrixXd> new_g;
	for (pair<string, MatrixXd> p: grads)
		new_g[p.first] = MatrixXd::Zero(p.second.rows(), p.second.cols());

	// "deltaOut" -- TODO ???? 
 	MatrixXd dY = outs; // - y;

 	new_g['dWV'] = h.transpose() * dY;
 	new_g['dBV'] = dY;
 	MatrixXd dh = dY * thetaV.transpose() + dh_next;

 	// dh/dout
 	MatrixXd dout = gradSigmoid(ho).array() * activationTanh(c).array() * dH.array*();

 	// dh/dc
 	MatrixXd dc = ho.array() * dh.array() * gradTanh(c).array();

 	// dc/dhf
 	MatrixXd dhf = gradSigmoid(hf).array() * c.array();

 	// dc/dhi
 	MatrixXd dhi = gradSigmoid(hi).array() * hc.array() * dc.array();

 	// dc/dhc
 	MatrixXd dhc = gradTanh(hc).array() * hi.array() * dc.array();

 	// gate grads

 	new_g['dWF'] = xPrime.transpose() * dhf;
 	new_g['dWIn'] = xPrime.transpose() * dhi;
 	new_g['dWOut'] = xPrime.transpose() * dho;
 	new_g['dWC'] = xPrime.transpose() * dhc;


 	new_g['dBF'] = dhf;
 	new_g['dBIn'] = dhi;
 	new_g['dBOut'] = dho;
 	new_g['dBC'] = dhc;

 	MatrixXd dX = dhf * thetaWForget.transpose() + 
 				  dhi * thetaWInput.transpose() + 
 				  dho * thetaWOutput.transpose() +
 				  dhc * thetaWCgate.transpose();

 	dh_next =  dX.leftCols(hiddenDim); // dx[:, :H]
 	dc_next = hf.array() * dc.array();

 	return make_tuple(new_g, dh_next, dc_next);



	// MatrixXd gradU = MatrixXd(thetaU.rows(), thetaU.cols());
	// MatrixXd gradW = MatrixXd(thetaW.rows(), thetaW.cols());
	// MatrixXd gradV = MatrixXd(thetaV.rows(), thetaV.cols());

	// int lastState = states.rows() - 1;

	// MatrixXd deltaT;

	// .array() * .array() --> elementwise multiplication
	// m * m --> matrix multiplication

	// for (int t = T-1; t >= 0; --t)
	// {
	// 	gradV += states.row(t).transpose() * deltaOut.row(t);

	// 	// initial DELTA that holds errors
	// 	deltaT = gradTanh(states.row(t > 0 ? t-1 : lastState)).array() *
	// 			(deltaOut.row(t) * thetaV.transpose()).array();		


	// 	for (int step = t; step >= max(0, t-bpttLen); --step)
	// 	{
	// 		gradW += states.row(step > 0 ? step - 1 : lastState).transpose() * 
	// 					deltaT;
	// 		gradU += x.row(t) * deltaT; // U = nxh, x.row(t) * U
	// 		deltaT = gradTanh(states.row(step > 0 ? step - 1 : lastState)).array()
	// 				* (deltaT * thetaW).array();
	// 	}
	// }

	// return tuple<map<string, MatrixXd, MatrixXd, MatrixXd>(upd_grads, dh_next, dc_next)
	// return boost::array<MatrixXd, 3>{{gradU, gradW, gradV}};
}


void 
RNN::trainSGD(MatrixXd &x, MatrixXd &y, 
			double alpha, int epoch, int checkLossEvery){

	vector<double> losses;
	int examples = y.rows();
	int currExample = 0;
	vector< tuple< MatrixXd, MatrixXd, MatrixXd, array<MatrixXd, 4>> > caches;


	grads = {
		{'dWIn', MatrixXd::Zero(thetaWInput.rows(), thetaWInput.cols())},
		{'dWOut', MatrixXd::Zero(thetaWOutput.rows(), thetaWOutput.cols())},
		{'dWF', MatrixXd::Zero(thetaWForget.rows(), thetaWForget.cols())},
		{'dWC', MatrixXd::Zero(thetaWCgate.rows(), thetaWCgate.cols())},
		{'dWV', MatrixXd::Zero(thetaV.rows(), thetaV.cols())},

		{'dBIn', MatrixXd::Zero(biasInput.rows(), biasInput.cols())},
		{'dBOut', MatrixXd::Zero(biasOut.rows(), biasOut.cols())},
		{'dBF', MatrixXd::Zero(biasForget.rows(), biasForget.cols())},
		{'dBC', MatrixXd::Zero(biasCgate.rows(), biasCgate.cols())},
		{'dBV', MatrixXd::Zero(biasV.rows(), biasV.cols())},
	};


	// TODO fprop

	cout << "Starting training..." << endl;
	for (int e = 0; e < epoch; ++e)
	{
		cout << "CURRENTLY AT EPOCH [" << e << "]..." << endl;

		if ((e % checkLossEvery) == 0 ){
			cout << "At iteration " << e << "..." << endl;
			double loss = cost(x, y, 0);
			cout << "Cost is currently..." << loss << endl;

			if (!losses.empty() && loss > losses.back()){
				// reduce learning rate in half if cost is increasing
				double newAlpha = alpha * 0.5;
				cout << "Changing learning rate from " << alpha << " to " << newAlpha << endl;
				alpha = newAlpha;

			}
			losses.push_back(loss);
		}

		for (int i = 0; i < examples; ++i)
		{
			// cout << "currently on example" << i << "..." << endl;

			auto a = backPropTT(x.row(i), y.row(i));

			thetaU -= alpha * dU;
			thetaW -= alpha * dW;
			thetaV -= alpha * dV;
		}
	}
}
