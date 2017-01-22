RNN::RNN(Eigen::MatrixXd m, int bpttLen = 30){

}

Eigen::MatrixXd RNN::getDateMatrix(); // first 5(? ish)
Eigen::MatrixXd RNN::getPricesMatrix(); // last n cols

Eigen::MatrixXd RNN::weightsRandomInit(int sizeR, int sizeR);

Eigen::MatrixXd RNN::activationFunction(Eigen::MatrixXd &m);
Eigen::MatrixXd RNN::activationGradientFunction(Eigen::MatrixXd &m);

vector<Eigen::MatrixXd> RNN::forwardProp(Eigen::MatrixXd x);
vector<Eigen::MatrixXd> RNN::backPropTT(Eigen::MatrixXd x, 
									Eigen::MatrixXd y);

Eigen::MatrixXd RNN::cost(Eigen::MatrixXd &theta,
					Eigen::MatrixXd &x,
					Eigen::MatrixXd &y,
					double reg_const);

Eigen::MatrixXd RNN::gradient(Eigen::MatrixXd &theta,
						Eigen::MatrixXd &x,
						Eigen::MatrixXd &y,
						double reg_const);

Eigen::MatrixXd RNN::predict(Eigen::MatrixXd &theta, 
						Eigen::MatrixXd &x_test);

Eigen::MatrixXd RNN::SGDStep();
Eigen::MatrixXd RNN::stochGradDesc(Eigen::MatrixXd &x_train,
							Eigen::MatrixXd &y_train,
							double alpha,
							int epoch,
							int check_loss_after=10);
void RNN::train(int iter = 10);

bool RNN::gradChecking(double error = 0.01);
