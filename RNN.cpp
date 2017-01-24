#include <iostream>
#include <Eigen/Dense>
#include <vector>	
#include "RNN.h"
#include <typeinfo>
#include <boost/array.hpp>

using namespace std;
using namespace Eigen;


/*
our RNN works as:
(INPUT) --> U(INPUT) --> W(U(INPUT)) [BPTT] --> V(W(U(INPUT))) = OUTPUT
*/

void print(auto s){ cout << s << endl; }


// data = MxN, M = # examples, N = 7
RNN::RNN(MatrixXd X, int bpttLen): 
featureDim(1), outputDim(1), hiddenDim(100), bpttLen(bpttLen), 
M(X.rows()), N(X.cols()), X(X)
{
	maxX = X.colwise().maxCoeff();
	minX = X.colwise().minCoeff();
	minMax = maxX - minX;

	// weights
	thetaU = MatrixXd::Random(featureDim, hiddenDim) * (1/sqrt(N));
	thetaV = MatrixXd::Random(hiddenDim, outputDim) * (1/sqrt(hiddenDim));
	thetaW = MatrixXd::Random(hiddenDim, hiddenDim) * (1/sqrt(hiddenDim));

}

// TODO: divide prices by one value
void RNN::normalize(MatrixXd &m){
	for (int i = 0; i < m.cols(); ++i)
		m.col(i) = (m.col(i).array() - minX(i)) / minMax(i);
	// m = (m.array() - minX) / (maxX - minX);
}

void RNN::denormalize(MatrixXd &m){
	for (int i = 0; i < m.cols(); ++i)
		m.col(i) = (m.col(i).array() * minMax(i)) + minX(i);
	// m = (m.array() * (maxX-minX)) + minX;

}

// tanh or reLU?
// reLU! f(x) = ln(1+e^x)
MatrixXd RNN::activationState(MatrixXd m){
	return (m.array().exp() + 1).log();
}

// softmax/logisitc sigmoid for now?
// f(t) = 1/ 1 + e^(-t)
MatrixXd RNN::activationOutput(MatrixXd m){
	return (((-1)*m).array().exp() + 1).array().inverse();
	
}

boost::array<MatrixXd, 2> RNN::forwardProp(MatrixXd x){
	int tSteps = x.rows(); // # examples

	// state matrix
	MatrixXd states = MatrixXd::Zero(tSteps + 1, hiddenDim);
	MatrixXd output = MatrixXd::Zero(tSteps, outputDim);

	for (int t = 0; t < tSteps; ++t)
	{
		// states.row(-1) == states.last_row()
		states.row(t) = activationState((x.row(t) * thetaU) + 
								(states.row(t > 0 ? t - 1: tSteps - 1) * thetaW));
		output.row(t) = activationOutput(states.row(t) * thetaV);
	}

	return boost::array<MatrixXd, 2>{{states, output}};
}

MatrixXd RNN::predict(MatrixXd &x_test){
	// since we are only predicting stock price, features out is Mx1
	return forwardProp(x_test)[1];
}


double RNN::cost(MatrixXd &x, MatrixXd &y, double reg_const){
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


	// TODO REGULARIZATION

	return loss;

}

boost::array<Eigen::MatrixXd, 3> RNN::backPropTT(MatrixXd x, MatrixXd y){

	int T = y.rows();
	// print(-5);
 	boost::array<MatrixXd, 2> a = forwardProp(x);
	// print(-4);
 	MatrixXd states = a[0];

	// deltaOut is the delta error/difference between output and actual
	MatrixXd deltaOut = a[1] - y;

	// graidents for weights
	MatrixXd gradU = MatrixXd(thetaU.rows(), thetaU.cols());
	MatrixXd gradW = MatrixXd(thetaW.rows(), thetaW.cols());
	MatrixXd gradV = MatrixXd(thetaV.rows(), thetaV.cols());

	// deltaOut[np.arange(len(y)), y] -= 1
	int lastState = states.rows() - 1;

	MatrixXd deltaT;
	// print(0);

	for (int t = T-1; t >= 0; --t)
	{
		gradV += states.row(t).transpose() * deltaOut.row(t);

		// initial DELTA that holds errors
		deltaT = (deltaOut.row(t) * thetaV.transpose()).array() 
				* (1- (states.row(t > 0 ? t-1 : lastState)
					.array().square()));

		for (int step = t; step >= max(0, t-bpttLen); --step)
		{
			gradW += states.row(step > 0 ? step - 1 : lastState).transpose() * deltaT;
			gradU += x.row(step).transpose() * deltaT;
			deltaT = (deltaT * thetaW).array() 
					* (1 - (states.row(step > 0 ? step -1 : lastState)
							.array().square()));
		}
	}

	return boost::array<MatrixXd, 3>{{gradU, gradW, gradV}};

}


void RNN::trainSGD(MatrixXd &x, MatrixXd &y, 
			double alpha, int epoch, int checkLossEvery){

	vector<double> losses;
	int examples = y.rows();
	int currExample = 0;

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

			boost::array<MatrixXd, 3> a = backPropTT(x.row(i), y.row(i));
			MatrixXd dU = a[0];
			MatrixXd dW = a[1];
			MatrixXd dV = a[2];

			thetaU -= alpha * dU;
			thetaW -= alpha * dW;
			thetaV -= alpha * dV;
		}
	}
}
